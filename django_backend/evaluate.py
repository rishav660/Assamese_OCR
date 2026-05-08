"""
Evaluate an Assamese OCR model on a labeled dataset.

Reports CER, WER, Exact Match, and shows sample predictions vs ground truth.

Usage:
    # Evaluate on validation set
    python evaluate.py --img-dir data/val_real_sentences/images \
                       --label-file data/val_real_sentences/labels/labels.txt

    # Evaluate on test set
    python evaluate.py --img-dir data/test_real_sentences/images \
                       --label-file data/test_real_sentences/labels/labels.txt

    # With spell-check post-processing
    python evaluate.py --img-dir data/val_real_sentences/images \
                       --label-file data/val_real_sentences/labels/labels.txt \
                       --post-process

    # Show more sample predictions
    python evaluate.py --img-dir data/val_real_sentences/images \
                       --label-file data/val_real_sentences/labels/labels.txt \
                       --num-samples 20
"""

import argparse
import os
import time

import cv2
import torch
from PIL import Image as PILImage
from torchvision import transforms

from char_map import char_to_idx, idx_to_char
from metrics import compute_cer, compute_wer, compute_exact_match, OCRMetrics
from model import AssameseOCR
from post_processing import correct_sentence
from beam_search import ctc_beam_search


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Assamese OCR model accuracy"
    )
    parser.add_argument(
        "--img-dir",
        type=str,
        required=True,
        help="Directory containing test/val images",
    )
    parser.add_argument(
        "--label-file",
        type=str,
        required=True,
        help="Labels manifest (TSV: filename<TAB>text)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model_sentences.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Image resize width",
    )
    parser.add_argument(
        "--post-process",
        action="store_true",
        help="Apply spell-check post-processing",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=1,
        help="Beam width for CTC decoding (1 = greedy, >1 = beam search)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of sample predictions to display",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional: save detailed results to a TSV file",
    )
    return parser.parse_args()


def load_labels(label_file):
    """Load labels from manifest file. Returns dict: filename -> text"""
    labels = {}
    with open(label_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                print(f"[WARNING] Line {line_num} malformed, skipping: {line[:50]}")
                continue
            filename, text = parts
            labels[os.path.basename(filename)] = text
    return labels


def load_model(checkpoint_path, device):
    model = AssameseOCR(img_height=64, nn_classes=len(char_to_idx) + 1)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def decode_single(log_probs):
    """Greedy CTC decode for a single output (T, C)."""
    blank_idx = len(char_to_idx)
    indices = torch.argmax(log_probs, dim=1)  # (T,)
    prev = -1
    chars = []
    for idx in indices:
        idx = idx.item()
        if idx != prev and idx != blank_idx:
            ch = idx_to_char.get(idx, "")
            if ch:
                chars.append(ch)
        prev = idx
    return "".join(chars)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    # Load labels
    labels = load_labels(args.label_file)
    print(f"Loaded {len(labels)} labels from {args.label_file}")

    # Find images that have labels
    valid_exts = {".jpg", ".jpeg", ".png"}
    all_images = sorted(
        f for f in os.listdir(args.img_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
    )
    matched = [f for f in all_images if f in labels]
    print(f"Found {len(all_images)} images, {len(matched)} with labels")

    if not matched:
        print("[ERROR] No image-label matches found. Check paths and filenames.")
        return

    from data_augmentation import AspectRatioResize
    # Transform
    transform = transforms.Compose([
        AspectRatioResize(target_height=64, max_width=2048),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # Run inference
    decode_method = f"beam search (width={args.beam_width})" if args.beam_width > 1 else "greedy"
    print(f"\nEvaluating {len(matched)} images...")
    print(f"Decode: {decode_method}")
    print(f"Post-processing: {'ON' if args.post_process else 'OFF'}\n")

    blank_idx = len(char_to_idx)

    metrics = OCRMetrics()
    all_predictions = []
    all_targets = []
    errors_detail = []  # (filename, prediction, ground_truth, cer)

    start_time = time.time()

    for i, filename in enumerate(matched):
        img_path = os.path.join(args.img_dir, filename)
        ground_truth = labels[filename]

        # Load and transform image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"  [SKIP] Could not read: {filename}")
            continue

        pil_image = PILImage.fromarray(image)
        tensor = transform(pil_image).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(tensor)
            outputs = torch.log_softmax(outputs, 2)

        if args.beam_width > 1:
            prediction = ctc_beam_search(
                outputs[:, 0, :].cpu(), idx_to_char, blank_idx,
                beam_width=args.beam_width,
            )
        else:
            prediction = decode_single(outputs[:, 0, :])  # greedy

        if args.post_process:
            prediction = correct_sentence(prediction)

        all_predictions.append(prediction)
        all_targets.append(ground_truth)

        # Per-sample CER for error analysis
        sample_cer = compute_cer([prediction], [ground_truth])
        errors_detail.append((filename, prediction, ground_truth, sample_cer))

        # Progress
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(matched)} images...")

    elapsed = time.time() - start_time
    metrics.update(all_predictions, all_targets)

    # === Results ===
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Dataset    : {args.img_dir} ({len(matched)} samples)")
    print(f"Decode     : {decode_method}")
    print(f"Post-proc  : {'ON' if args.post_process else 'OFF'}")
    print(f"Time       : {elapsed:.1f}s ({len(matched) / elapsed:.1f} img/s)")
    print("-" * 70)
    print(f"CER         : {metrics.cer:.4f}  ({metrics.cer * 100:.2f}%)")
    print(f"WER         : {metrics.wer:.4f}  ({metrics.wer * 100:.2f}%)")
    print(f"Exact Match : {metrics.exact_match:.4f}  ({metrics.exact_match * 100:.2f}%)")
    print("=" * 70)

    # Show sample predictions (mix of best and worst)
    errors_detail.sort(key=lambda x: x[3])  # sort by CER

    print(f"\n--- Best {min(args.num_samples, len(errors_detail))} predictions ---")
    for fname, pred, gt, cer in errors_detail[: args.num_samples]:
        status = "✅" if pred == gt else "❌"
        print(f"  {status} [{fname}] CER={cer:.2%}")
        print(f"     GT  : {gt}")
        print(f"     Pred: {pred}")

    worst = errors_detail[-args.num_samples:]
    worst.reverse()
    print(f"\n--- Worst {len(worst)} predictions ---")
    for fname, pred, gt, cer in worst:
        status = "✅" if pred == gt else "❌"
        print(f"  {status} [{fname}] CER={cer:.2%}")
        print(f"     GT  : {gt}")
        print(f"     Pred: {pred}")

    # Optional: save full results to TSV
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write("filename\tground_truth\tprediction\tcer\texact_match\n")
            for fname, pred, gt, cer in errors_detail:
                exact = "1" if pred == gt else "0"
                f.write(f"{fname}\t{gt}\t{pred}\t{cer:.6f}\t{exact}\n")
        print(f"\nDetailed results saved to: {args.output}")

    return metrics


if __name__ == "__main__":
    evaluate(parse_args())

"""
Terminal-first OCR prediction helper for sentence-level models.
"""

import argparse

import cv2
import torch
from PIL import Image as PILImage
from torchvision import transforms

from char_map import char_to_idx, idx_to_char
from model import AssameseOCR
from post_processing import correct_sentence
from metrics import compute_cer, compute_wer
from beam_search import ctc_beam_search


def parse_args():
    parser = argparse.ArgumentParser(description="Run OCR prediction from the terminal")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model_sentences.pth",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Resize width for the OCR model",
    )
    parser.add_argument(
        "--disable-post-process",
        action="store_true",
        help="Skip spell correction post-processing",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=1,
        help="Beam width for CTC decoding (1 = greedy, >1 = beam search)",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="Ground-truth text (or path to a .txt file) for CER/WER evaluation",
    )
    return parser.parse_args()


def decode_prediction(preds, pred_sizes):
    decoded_texts = []
    preds = preds.permute(1, 0, 2).cpu()
    preds = torch.argmax(preds, dim=2)

    for i in range(preds.size(0)):
        pred_seq = preds[i][: pred_sizes[i]]
        prev_char = -1
        text = ""
        for idx in pred_seq:
            idx = idx.item()
            if idx != prev_char and idx != len(char_to_idx):
                text += idx_to_char.get(idx, "")
            prev_char = idx
        decoded_texts.append(text)
    return decoded_texts


def load_model(checkpoint_path, device):
    model = AssameseOCR(img_height=64, nn_classes=len(char_to_idx) + 1)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict(image_path, checkpoint_path, width=512, use_post_process=True,
            beam_width=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)

    from data_augmentation import AspectRatioResize
    transform = transforms.Compose(
        [
            AspectRatioResize(target_height=64, max_width=2048),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    pil_image = PILImage.fromarray(image)
    image_tensor = transform(pil_image).unsqueeze(0).to(device)

    blank_idx = len(char_to_idx)

    with torch.no_grad():
        outputs = model(image_tensor)
        outputs = torch.log_softmax(outputs, 2)

        if beam_width > 1:
            # Beam search decode
            text = ctc_beam_search(
                outputs[:, 0, :].cpu(), idx_to_char, blank_idx,
                beam_width=beam_width,
            )
        else:
            # Greedy decode
            pred_sizes = torch.full(
                size=(outputs.size(1),),
                fill_value=outputs.size(0),
                dtype=torch.int32,
            )
            decoded = decode_prediction(outputs, pred_sizes)
            text = decoded[0]

    if use_post_process:
        text = correct_sentence(text)
    return text


def main():
    args = parse_args()
    text = predict(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        width=args.width,
        use_post_process=not args.disable_post_process,
        beam_width=args.beam_width,
    )
    decode_method = f"beam search (width={args.beam_width})" if args.beam_width > 1 else "greedy"
    print(f"Decode: {decode_method}")
    print(f"Prediction: {text}")

    # Evaluate against ground truth if provided
    if args.ground_truth:
        gt = args.ground_truth
        # If it looks like a file path, read from it
        if gt.endswith(".txt") and os.path.isfile(gt):
            with open(gt, "r", encoding="utf-8") as f:
                gt = f.read().strip()

        cer = compute_cer([text], [gt])
        wer = compute_wer([text], [gt])
        exact = "✅ EXACT MATCH" if text == gt else "❌ Not exact"

        print(f"Ground Truth: {gt}")
        print(f"CER: {cer:.4f} ({cer * 100:.2f}%)")
        print(f"WER: {wer:.4f} ({wer * 100:.2f}%)")
        print(exact)


if __name__ == "__main__":
    import os
    main()

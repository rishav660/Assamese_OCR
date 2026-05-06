"""
Train a sentence-level Assamese OCR model from real sentence data.

This version keeps the original training logic but adds CLI arguments so the
same script is easier to use in Colab or a training-only repo.
"""

import argparse
import multiprocessing
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from char_map import char_to_idx, idx_to_char
from data_augmentation import get_training_transforms, get_validation_transforms
from dataset import AssameseOCRDataset, collate_fn
from metrics import OCRMetrics
from model import AssameseOCR


def default_num_workers():
    return 2 if os.name == "nt" else 4


def decode_ctc_greedy(log_probs, input_lengths, labels, target_lengths):
    """
    Decode CTC outputs with greedy search and reconstruct target strings.

    Args:
        log_probs:      (T, B, C) log-softmax output from the model
        input_lengths:  (B,) tensor of sequence lengths
        labels:         (sum(target_lengths),) flattened target indices
        target_lengths: (B,) tensor of target lengths per sample

    Returns:
        predictions: list[str] — decoded predictions
        targets:     list[str] — ground-truth strings
    """
    blank_idx = len(char_to_idx)
    batch_size = log_probs.size(1)

    # --- decode predictions ---
    preds = log_probs.permute(1, 0, 2).cpu()       # (B, T, C)
    preds = torch.argmax(preds, dim=2)              # (B, T)

    predictions = []
    for i in range(batch_size):
        seq = preds[i][: input_lengths[i]]
        prev = -1
        chars = []
        for idx in seq:
            idx = idx.item()
            if idx != prev and idx != blank_idx:
                ch = idx_to_char.get(idx, "")
                if ch:
                    chars.append(ch)
            prev = idx
        predictions.append("".join(chars))

    # --- reconstruct targets ---
    targets = []
    offset = 0
    for i in range(batch_size):
        length = target_lengths[i].item()
        indices = labels[offset : offset + length].cpu().tolist()
        target_str = "".join(idx_to_char.get(idx, "") for idx in indices)
        targets.append(target_str)
        offset += length

    return predictions, targets


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a sentence-level Assamese OCR model"
    )
    parser.add_argument(
        "--train-img-dir",
        type=str,
        default="data/train_real_sentences/images",
        help="Training image directory",
    )
    parser.add_argument(
        "--train-label-file",
        type=str,
        default="data/train_real_sentences/labels/labels.txt",
        help="Training labels manifest",
    )
    parser.add_argument(
        "--val-img-dir",
        type=str,
        default="data/val_real_sentences/images",
        help="Validation image directory",
    )
    parser.add_argument(
        "--val-label-file",
        type=str,
        default="data/val_real_sentences/labels/labels.txt",
        help="Validation labels manifest",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Maximum number of epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0001,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay",
    )
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=3,
        help="ReduceLROnPlateau patience",
    )
    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.5,
        help="ReduceLROnPlateau decay factor",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=5,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=default_num_workers(),
        help="DataLoader worker count",
    )
    parser.add_argument(
        "--disable-augmentation",
        action="store_true",
        help="Disable OCR-specific training augmentation",
    )
    parser.add_argument(
        "--best-checkpoint",
        type=str,
        default="checkpoints/best_model_sentences.pth",
        help="Path for best checkpoint",
    )
    parser.add_argument(
        "--final-checkpoint",
        type=str,
        default="checkpoints/final_model_sentences.pth",
        help="Path for final checkpoint",
    )
    parser.add_argument(
        "--plot-out",
        type=str,
        default="training_curve_sentences.png",
        help="Path for loss plot image",
    )
    return parser.parse_args()


def ensure_parent_dir(path_str):
    path = Path(path_str)
    if path.parent and str(path.parent) != ".":
        path.parent.mkdir(parents=True, exist_ok=True)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if " " in char_to_idx:
        print(f"Space character found in char_map at index: {char_to_idx[' ']}")
    else:
        print("WARNING: space character is missing from char_map.")

    print("\nLoading sentence datasets...")
    print(f"  train images : {args.train_img_dir}")
    print(f"  train labels : {args.train_label_file}")
    print(f"  val images   : {args.val_img_dir}")
    print(f"  val labels   : {args.val_label_file}")

    train_transform = get_training_transforms(augment=not args.disable_augmentation)
    val_transform = get_validation_transforms()

    print(
        f"Data augmentation: {'disabled' if args.disable_augmentation else 'enabled'}"
    )

    train_dataset = AssameseOCRDataset(
        img_dir=args.train_img_dir,
        label_file=args.train_label_file,
        char_to_idx=char_to_idx,
        transform=train_transform,
    )

    val_dataset = AssameseOCRDataset(
        img_dir=args.val_img_dir,
        label_file=args.val_label_file,
        char_to_idx=char_to_idx,
        transform=val_transform,
    )

    print(f"Train sentences: {len(train_dataset)}, Val sentences: {len(val_dataset)}")

    if len(train_dataset) == 0:
        raise RuntimeError(
            "No training data found. Build train/val splits before starting training."
        )
    if len(val_dataset) == 0:
        raise RuntimeError(
            "No validation data found. Build a validation split before starting training."
        )

    pin_memory = torch.cuda.is_available()
    print(f"Using {args.num_workers} workers for data loading")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    num_classes = len(char_to_idx) + 1
    model = AssameseOCR(img_height=64, nn_classes=num_classes).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CTCLoss(
        blank=len(char_to_idx),
        reduction="mean",
        zero_infinity=True,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        patience=args.scheduler_patience,
        factor=args.scheduler_factor,
    )

    print("\n" + "=" * 60)
    print("Sentence OCR training")
    print("=" * 60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Weight decay: {args.weight_decay}")
    print("=" * 60 + "\n")

    print("Loading first batch for sanity check...")
    sample_batch = next(iter(train_loader))
    if sample_batch is not None:
        images, labels, input_lengths, target_lengths = sample_batch
        print(f"Batch loaded: {images.shape}")
        print(f"Sequence length: {input_lengths[0].item()}")
        print(f"Sample target lengths: {target_lengths[:4].tolist()}")
        space_idx = char_to_idx.get(" ", -1)
        print(f"Labels contain spaces: {space_idx > 0 and space_idx in labels}")

    best_val_loss = float("inf")
    best_cer = float("inf")
    train_losses = []
    val_losses = []
    val_cers = []
    val_wers = []
    no_improve_epochs = 0

    ensure_parent_dir(args.best_checkpoint)
    ensure_parent_dir(args.final_checkpoint)
    ensure_parent_dir(args.plot_out)

    print("\nStarting training...\n")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue

            images, labels, input_lengths, target_lengths = batch
            images = images.to(device)
            labels = labels.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

            outputs = model(images)
            outputs = torch.log_softmax(outputs, 2)
            loss = criterion(outputs, labels, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            if batch_idx % 50 == 0:
                elapsed = time.time() - epoch_start
                batches_per_sec = (batch_idx + 1) / elapsed if elapsed > 0 else 0
                eta = (
                    (len(train_loader) - batch_idx) / batches_per_sec
                    if batches_per_sec > 0
                    else 0
                )
                print(
                    f"Epoch {epoch + 1}/{args.epochs} | "
                    f"Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Speed: {batches_per_sec:.2f} batch/s | ETA: {eta:.0f}s"
                )

        avg_train_loss = total_loss / batch_count if batch_count > 0 else 0.0
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        val_batch_count = 0
        val_metrics = OCRMetrics()

        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue

                images, labels, input_lengths, target_lengths = batch
                images = images.to(device)
                labels = labels.to(device)
                input_lengths = input_lengths.to(device)
                target_lengths = target_lengths.to(device)

                outputs = model(images)
                outputs = torch.log_softmax(outputs, 2)
                loss = criterion(outputs, labels, input_lengths, target_lengths)

                val_loss += loss.item()
                val_batch_count += 1

                # Decode predictions and compute CER / WER
                predictions, targets = decode_ctc_greedy(
                    outputs, input_lengths, labels, target_lengths
                )
                val_metrics.update(predictions, targets)

        avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else 0.0
        val_losses.append(avg_val_loss)
        val_cers.append(val_metrics.cer)
        val_wers.append(val_metrics.wer)
        epoch_time = time.time() - epoch_start

        print(f"\n{'=' * 60}")
        print(
            f"Epoch {epoch + 1}/{args.epochs} completed in {epoch_time:.1f}s "
            f"({epoch_time / 60:.1f} min)"
        )
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val {val_metrics.summary()}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_cer = val_metrics.cer
            torch.save(model.state_dict(), args.best_checkpoint)
            print(f"Best model saved to: {args.best_checkpoint}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epoch(s)")

        print(f"{'=' * 60}\n")

        scheduler.step(avg_val_loss)

        if no_improve_epochs >= args.early_stop_patience:
            print(f"Early stopping after {epoch + 1} epochs")
            break

    torch.save(model.state_dict(), args.final_checkpoint)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Loss plot
    ax1.plot(train_losses, label="Training Loss")
    ax1.plot(val_losses, label="Validation Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Loss")

    # CER / WER plot
    ax2.plot(val_cers, label="CER", marker="o", markersize=3)
    ax2.plot(val_wers, label="WER", marker="s", markersize=3)
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Error Rate")
    ax2.legend()
    ax2.set_title("Character & Word Error Rate")

    fig.suptitle("Sentence Model Training", fontsize=14)
    fig.tight_layout()
    fig.savefig(args.plot_out)

    print("\n" + "=" * 60)
    print("Training complete")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best CER at that point: {best_cer:.4f} ({best_cer * 100:.2f}%)")
    print(f"Final CER: {val_cers[-1]:.4f} ({val_cers[-1] * 100:.2f}%)" if val_cers else "")
    print(f"Final WER: {val_wers[-1]:.4f} ({val_wers[-1] * 100:.2f}%)" if val_wers else "")
    print(f"Best checkpoint: {args.best_checkpoint}")
    print(f"Final checkpoint: {args.final_checkpoint}")
    print(f"Loss plot: {args.plot_out}")
    print("=" * 60)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    train(parse_args())

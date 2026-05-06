"""
Build non-overlapping train/validation/test splits for real Assamese sentence OCR.

This script reads the corpus once, deduplicates valid Assamese sentences,
shuffles them with a fixed seed, and writes separate image/label splits.
"""

import argparse
import random
import re
from pathlib import Path

from generate_real_sentence_data import RealSentenceGenerator

ASSAMESE_RE = re.compile(r"[\u0980-\u09FF]")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create clean, non-overlapping sentence OCR splits"
    )
    parser.add_argument("--input", type=str, default="data/as-wiki-2021.txt", help="Input corpus file")
    parser.add_argument("--train-output", type=str, default="data/train_real_sentences", help="Output directory for training split")
    parser.add_argument("--val-output", type=str, default="data/val_real_sentences", help="Output directory for validation split")
    parser.add_argument("--test-output", type=str, default="data/test_real_sentences", help="Output directory for optional test split")
    parser.add_argument("--train-count", type=int, default=15000, help="Number of training sentences to generate")
    parser.add_argument("--val-count", type=int, default=4000, help="Number of validation sentences to generate")
    parser.add_argument("--test-count", type=int, default=0, help="Number of optional test sentences to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--font", type=str, default="fonts/NotoSansBengali-Regular.ttf", help="Fallback font path")
    parser.add_argument("--font-size", type=int, default=48, help="Font size for rendered sentences")
    parser.add_argument("--img-height", type=int, default=64, help="Minimum rendered image height")
    parser.add_argument("--min-length", type=int, default=5, help="Minimum sentence length")
    parser.add_argument("--max-length", type=int, default=150, help="Maximum sentence length")
    parser.add_argument("--disable-multi-font", action="store_true", help="Use a single font instead of the working font list")
    return parser.parse_args()


def collect_sentences(input_file, min_length, max_length):
    seen = set()
    sentences = []
    stats = {
        "total_lines": 0,
        "blank": 0,
        "too_short": 0,
        "too_long": 0,
        "non_assamese": 0,
        "duplicates": 0,
    }

    with open(input_file, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            stats["total_lines"] += 1
            sentence = raw_line.strip()
            sentence = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]", "", sentence)
            sentence = re.sub(r"\s+", " ", sentence)

            if not sentence:
                stats["blank"] += 1
                continue
            if len(sentence) < min_length:
                stats["too_short"] += 1
                continue
            if len(sentence) > max_length:
                stats["too_long"] += 1
                continue
            if not ASSAMESE_RE.search(sentence):
                stats["non_assamese"] += 1
                continue
            if sentence in seen:
                stats["duplicates"] += 1
                continue

            seen.add(sentence)
            sentences.append(sentence)

    return sentences, stats


import multiprocessing
from functools import partial

def _process_single_sentence(args):
    offset, sentence, output_dir_str, start_index, font_paths, font_size, img_height, padding = args
    from pathlib import Path
    from generate_real_sentence_data import RealSentenceGenerator
    
    # We create a lightweight generator just for rendering (no file I/O overhead)
    # The font path list is passed so it can randomly choose
    generator = RealSentenceGenerator(
        input_file="", output_dir="", font_path=None, 
        font_size=font_size, img_height=img_height, padding=padding, use_multiple_fonts=False
    )
    # Override font paths for random selection
    generator.font_paths = font_paths
    
    output_path = Path(output_dir_str)
    image_dir = output_path / "images"
    label_dir = output_path / "labels"
    
    file_index = start_index + offset
    image_name = f"sentence_{file_index:06d}.png"
    image_path = image_dir / image_name

    image = generator.render_sentence_image(sentence)
    image.save(image_path)

    label_file = label_dir / f"sentence_{file_index:06d}.txt"
    with open(label_file, "w", encoding="utf-8") as text_handle:
        text_handle.write(sentence)
        
    return f"{image_name}\t{sentence}\n"

def write_split(sentences, output_dir, generator, start_index=0):
    output_path = Path(output_dir)
    image_dir = output_path / "images"
    label_dir = output_path / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = label_dir / "labels.txt"
    
    # Prepare arguments for multiprocessing
    total = len(sentences)
    args_list = [
        (i, sentence, str(output_dir), start_index, generator.font_paths, generator.font_size, generator.img_height, generator.padding) 
        for i, sentence in enumerate(sentences)
    ]
    
    print(f"  -> Generating {total} images using {multiprocessing.cpu_count()} CPU cores...")
    
    with open(manifest_path, "w", encoding="utf-8") as manifest:
        # Use multiprocessing pool
        with multiprocessing.Pool() as pool:
            for i, result_line in enumerate(pool.imap_unordered(_process_single_sentence, args_list)):
                manifest.write(result_line)
                if (i + 1) % 1000 == 0 or (i + 1) == total:
                    print(f"     Progress: {i + 1} / {total} ({(i + 1)/total*100:.1f}%)")

    return manifest_path


def main():
    args = parse_args()

    generator = RealSentenceGenerator(
        input_file=args.input,
        output_dir=args.train_output,
        font_path=args.font,
        font_size=args.font_size,
        img_height=args.img_height,
        min_length=args.min_length,
        max_length=args.max_length,
        use_multiple_fonts=not args.disable_multi_font,
    )

    print("=" * 72)
    print("Building real sentence splits")
    print("=" * 72)
    print(f"Corpus: {args.input}")
    print(f"Seed: {args.seed}")
    print(f"Requested splits: train={args.train_count}, val={args.val_count}, test={args.test_count}")

    all_sentences, stats = collect_sentences(
        input_file=args.input,
        min_length=args.min_length,
        max_length=args.max_length,
    )

    print(f"Unique valid sentences available: {len(all_sentences)}")
    print(
        "Filtered out: "
        f"blank={stats['blank']}, too_short={stats['too_short']}, "
        f"too_long={stats['too_long']}, non_assamese={stats['non_assamese']}, "
        f"duplicates={stats['duplicates']}"
    )

    needed = args.train_count + args.val_count + args.test_count
    if needed > len(all_sentences):
        raise ValueError(
            f"Requested {needed} sentences but only {len(all_sentences)} are available."
        )

    rng = random.Random(args.seed)
    rng.shuffle(all_sentences)

    train_sentences = all_sentences[: args.train_count]
    val_start = args.train_count
    val_end = val_start + args.val_count
    val_sentences = all_sentences[val_start:val_end]
    test_sentences = all_sentences[val_end : val_end + args.test_count]

    train_set = set(train_sentences)
    val_set = set(val_sentences)
    test_set = set(test_sentences)

    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("Split overlap detected. Aborting.")

    print("Writing training split...")
    train_manifest = write_split(train_sentences, args.train_output, generator, start_index=0)
    print("Writing validation split...")
    val_manifest = write_split(val_sentences, args.val_output, generator, start_index=0)

    test_manifest = None
    if args.test_count:
        print("Writing test split...")
        test_manifest = write_split(test_sentences, args.test_output, generator, start_index=0)

    print("=" * 72)
    print("Done")
    print("=" * 72)
    print(f"Train: {len(train_sentences)} -> {train_manifest}")
    print(f"Val:   {len(val_sentences)} -> {val_manifest}")
    if test_manifest:
        print(f"Test:  {len(test_sentences)} -> {test_manifest}")
    print("Overlap check: PASSED")


if __name__ == "__main__":
    main()

"""
Real Sentence Dataset Generator
Generates sentence-level training data from real Assamese sentences
Uses proper font rendering instead of synthetic word combinations
"""
import os
import argparse
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Set

class RealSentenceGenerator:
    """Generate sentence images from real Assamese text"""
    
    def __init__(
        self,
        input_file: str,
        output_dir: str,
        font_path: str = None,  # Can be a single font or None to use multiple
        font_size: int = 48,
        img_height: int = 64,
        min_length: int = 5,
        max_length: int = 150,
        padding: int = 10,
        use_multiple_fonts: bool = True  # NEW: Use multiple fonts
    ):
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.font_size = font_size
        self.img_height = img_height
        self.min_length = min_length
        self.max_length = max_length
        self.padding = padding
        self.use_multiple_fonts = use_multiple_fonts
        
        # Load available fonts
        if use_multiple_fonts and os.path.exists("fonts/working_fonts.txt"):
            with open("fonts/working_fonts.txt", 'r') as f:
                self.font_paths = [line.strip() for line in f if line.strip()]
            print(f"✓ Loaded {len(self.font_paths)} fonts for variety")
        elif font_path:
            self.font_paths = [font_path]
            print(f"✓ Using single font: {font_path}")
        else:
            # Default fallback
            self.font_paths = ["fonts/NotoSansBengali-Regular.ttf"]
            print(f"✓ Using default font")
        
        # Statistics
        
        # Statistics
        self.stats = {
            'total_lines': 0,
            'processed': 0,
            'skipped_short': 0,
            'skipped_long': 0,
            'skipped_invalid': 0,
            'duplicates': 0,
            'successful': 0,
            'errors': 0
        }
        
        # Track seen sentences to avoid duplicates
        self.seen_sentences: Set[str] = set()
        
    def clean_sentence(self, text: str) -> str:
        """Clean and normalize sentence text"""
        # Strip whitespace
        text = text.strip()
        
        # Remove control characters except newline
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Normalize multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def is_valid_sentence(self, text: str) -> bool:
        """Check if sentence is valid for dataset"""
        # Check length
        if len(text) < self.min_length:
            self.stats['skipped_short'] += 1
            return False
        
        if len(text) > self.max_length:
            self.stats['skipped_long'] += 1
            return False
        
        # Check for Assamese Unicode characters (Bengali script range)
        # Assamese uses Bengali script: U+0980 to U+09FF
        has_assamese = bool(re.search(r'[\u0980-\u09FF]', text))
        if not has_assamese:
            self.stats['skipped_invalid'] += 1
            return False
        
        # Check for duplicates
        if text in self.seen_sentences:
            self.stats['duplicates'] += 1
            return False
        
        return True
    
    def render_sentence_image(self, text: str) -> Image.Image:
        """Render sentence as an image with proper font"""
        # Randomly select a font for variety
        import random
        font_path = random.choice(self.font_paths)
        
        try:
            # Load font
            font = ImageFont.truetype(font_path, self.font_size)
        except Exception as e:
            print(f"Warning: Could not load font {font_path}: {e}")
            print("Using default font...")
            font = ImageFont.load_default()
        
        # Create a dummy image to measure text size
        dummy_img = Image.new('L', (1, 1), color=255)
        draw = ImageDraw.Draw(dummy_img)
        
        # Get text bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Calculate image dimensions with padding
        img_width = text_width + (self.padding * 2)
        img_height = max(self.img_height, text_height + (self.padding * 2))
        
        # Create actual image
        img = Image.new('L', (img_width, img_height), color=255)  # White background
        draw = ImageDraw.Draw(img)
        
        # Calculate text position (centered vertically)
        x = self.padding
        y = (img_height - text_height) // 2
        
        # Draw text
        draw.text((x, y), text, fill=0, font=font)  # Black text
        
        return img
    
    def generate_dataset(self, num_sentences: int = None, start_index: int = 0):
        """Generate dataset from input file"""
        # Create output directories
        img_dir = self.output_dir / 'images'
        label_dir = self.output_dir / 'labels'
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"{'='*70}")
        print(f"Real Sentence Dataset Generator")
        print(f"{'='*70}")
        print(f"Input file: {self.input_file}")
        print(f"Output directory: {self.output_dir}")
        if len(self.font_paths) > 1:
            print(f"Fonts: {len(self.font_paths)} fonts (random selection)")
        else:
            print(f"Font: {self.font_paths[0]}")
        print(f"{'='*70}\n")
        
        # Open manifest file
        manifest_path = label_dir / 'labels.txt'
        manifest_file = open(manifest_path, 'w', encoding='utf-8')
        
        # Read and process sentences
        print("Reading sentences...")
        
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                sentence_count = 0
                
                for line in f:
                    self.stats['total_lines'] += 1
                    
                    # Skip if we've reached the desired number
                    if num_sentences and sentence_count >= num_sentences:
                        break
                    
                    # Clean sentence
                    sentence = self.clean_sentence(line)
                    
                    if not sentence:
                        continue
                    
                    self.stats['processed'] += 1
                    
                    # Validate sentence
                    if not self.is_valid_sentence(sentence):
                        continue
                    
                    # Track this sentence
                    self.seen_sentences.add(sentence)
                    
                    # Generate output filename
                    idx = start_index + sentence_count
                    filename = f"sentence_{idx:06d}"
                    img_filename = f"{filename}.png"
                    txt_filename = f"{filename}.txt"
                    
                    try:
                        # Render image
                        img = self.render_sentence_image(sentence)
                        
                        # Save image
                        img_path = img_dir / img_filename
                        img.save(img_path)
                        
                        # Save text file
                        txt_path = label_dir / txt_filename
                        with open(txt_path, 'w', encoding='utf-8') as txt_file:
                            txt_file.write(sentence)
                        
                        # Write to manifest
                        manifest_file.write(f"{img_filename}\t{sentence}\n")
                        
                        self.stats['successful'] += 1
                        sentence_count += 1
                        
                        # Progress update
                        if sentence_count % 100 == 0:
                            print(f"  Generated {sentence_count} sentences...")
                    
                    except Exception as e:
                        self.stats['errors'] += 1
                        print(f"  Error processing sentence {idx}: {e}")
                        continue
        
        finally:
            manifest_file.close()
        
        # Print statistics
        print(f"\n{'='*70}")
        print(f"Generation Complete!")
        print(f"{'='*70}")
        print(f"Total lines read: {self.stats['total_lines']}")
        print(f"Lines processed: {self.stats['processed']}")
        print(f"Successfully generated: {self.stats['successful']}")
        print(f"\nSkipped:")
        print(f"  - Too short: {self.stats['skipped_short']}")
        print(f"  - Too long: {self.stats['skipped_long']}")
        print(f"  - Invalid (no Assamese): {self.stats['skipped_invalid']}")
        print(f"  - Duplicates: {self.stats['duplicates']}")
        print(f"  - Errors: {self.stats['errors']}")
        print(f"\nOutput:")
        print(f"  - Images: {img_dir}")
        print(f"  - Labels: {label_dir}")
        print(f"  - Manifest: {manifest_path}")
        print(f"{'='*70}")
        
        return self.stats['successful']


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Generate sentence dataset from real Assamese text'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/as-wiki-2021.txt',
        help='Input text file with Assamese sentences'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/real_sentences',
        help='Output directory for generated dataset'
    )
    parser.add_argument(
        '--font',
        type=str,
        default='fonts/NotoSansBengali-Regular.ttf',
        help='Path to Assamese/Bengali font file'
    )
    parser.add_argument(
        '--num',
        type=int,
        default=None,
        help='Number of sentences to generate (default: all)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: generate only 10 sentences'
    )
    parser.add_argument(
        '--font-size',
        type=int,
        default=48,
        help='Font size for rendering (default: 48)'
    )
    parser.add_argument(
        '--img-height',
        type=int,
        default=64,
        help='Minimum image height (default: 64)'
    )
    
    args = parser.parse_args()
    
    # Test mode override
    if args.test:
        args.num = 10
        print("\n*** TEST MODE: Generating 10 sentences ***\n")
    
    # Create generator
    generator = RealSentenceGenerator(
        input_file=args.input,
        output_dir=args.output,
        font_path=args.font,
        font_size=args.font_size,
        img_height=args.img_height
    )
    
    # Generate dataset
    count = generator.generate_dataset(num_sentences=args.num)
    
    print(f"\n✓ Successfully generated {count} sentence images!")
    
    if args.test:
        print(f"\nTest complete! Check the output at: {args.output}")
        print("If everything looks good, run without --test to generate full dataset.")


if __name__ == "__main__":
    main()

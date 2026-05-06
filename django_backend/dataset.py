import os
import torch
import random
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class AssameseOCRDataset(Dataset):
    def __init__(self, img_dir, label_file, char_to_idx, transform=None, max_images=None):
        self.img_dir = img_dir
        self.char_to_idx = char_to_idx
        self.transform = transform
        self.max_images = max_images

        self.image_files = []
        self.labels = {}

        if not os.path.exists(img_dir):
            print(f"[ERROR] Image directory '{img_dir}' does not exist.")
            return
        if not os.path.exists(label_file):
            print(f"[ERROR] Label file '{label_file}' does not exist.")
            return

        print(f"Loading labels from {label_file}...")
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        parts = line.strip().split('\t')
                        if len(parts) != 2:
                            print(f"[WARNING] Line {line_num} is malformed: {line.strip()}")
                            continue
                        filename, text = parts
                        filename = os.path.basename(filename)
                        base_name = os.path.splitext(filename)[0]
                        text = text.replace('\u200c', '').replace('\t', '')  # Clean text but keep spaces
                        text = text[:150]  # Increase limit for sentences

                        self.labels[filename] = text
                        self.labels[base_name] = text
                        if not any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                            self.labels[f"{filename}.jpeg"] = text
                            self.labels[f"{base_name}.jpeg"] = text
        except Exception as e:
            print(f"[ERROR] Failed to read label file: {e}")
            return

        print(f"Loaded {len(set(self.labels.values()))} unique labels.")

        valid_extensions = ['.jpg', '.jpeg', '.png']
        all_files = os.listdir(img_dir)
        potential_images = [f for f in all_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

        print(f"Found {len(potential_images)} image files in directory.")

        for img_file in potential_images:
            base_name = os.path.splitext(img_file)[0]
            if img_file in self.labels or base_name in self.labels:
                self.image_files.append(img_file)

        print(f"Matched {len(self.image_files)} images with labels.")

        if self.max_images is not None and len(self.image_files) > self.max_images:
            print(f"Randomly selecting {self.max_images} images from {len(self.image_files)} available")
            random.shuffle(self.image_files)
            self.image_files = self.image_files[:self.max_images]
            print(f"Dataset limited to {len(self.image_files)} images")

        if not self.image_files:
            print("\n[WARNING] No image-label matches found. Check:")
            print(" - File names in label file and image folder")
            print(" - UTF-8 encoding and label file format (filename<TAB>label)")
            print(" - Presence of supported image extensions (.jpg, .jpeg, .png)")
            print(f"\nSample image files: {potential_images[:5]}")
            print(f"Sample label keys: {list(self.labels.keys())[:5]}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_file)

        label = self.labels.get(img_file)
        if label is None:
            base_name = os.path.splitext(img_file)[0]
            label = self.labels.get(base_name)

        if not label:
            print(f"[WARNING] Skipping {img_file} due to missing label.")
            return None, None

        try:
            image = Image.open(img_path).convert('L')
        except Exception as e:
            print(f"[ERROR] Failed to load image '{img_path}': {e}")
            return None, None

        if self.transform:
            image = self.transform(image)

        if image is None:
            print(f"[ERROR] Invalid image: {img_path}")
            return None, None

        if torch.isnan(image).any() or torch.isinf(image).any():
            print(f"[ERROR] Invalid image tensor: {img_path}")
            return None, None

        try:
            label_encoded = [self.char_to_idx[c] for c in label if c in self.char_to_idx]
        except KeyError as e:
            print(f"[ERROR] Unknown character '{e.args[0]}' in label for '{img_file}'")
            return None, None

        if len(label_encoded) == 0:
            print(f"[WARNING] Skipping {img_file} due to no valid characters in label: {label}")
            return None, None

        return image, torch.tensor(label_encoded, dtype=torch.long)

def collate_fn(batch):
    batch = [b for b in batch if b is not None and len(b[1]) > 0]
    if len(batch) == 0:
        return None, None, None, None

    images, labels = zip(*batch)
    
    # Pad images to max width in this batch
    max_w = max(img.shape[2] for img in images)
    
    padded_images = []
    for img in images:
        c, h, w = img.shape
        pad_w = max_w - w
        # Pad on the right
        padded_img = torch.nn.functional.pad(img, (0, pad_w, 0, 0), value=0)
        padded_images.append(padded_img)
        
    images = torch.stack(padded_images)

    # Flatten labels
    flattened_labels = torch.cat(labels)
    target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)

    # The CNN downsamples width by exactly 4. 
    # Our AspectRatioResize ensures widths are divisible by 4, so this is exact.
    input_lengths = torch.tensor([img.shape[2] // 4 for img in padded_images], dtype=torch.long)

    return images, flattened_labels, input_lengths, target_lengths

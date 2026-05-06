""" 
Data Augmentation for OCR Training
Adds variations to make model robust to different fonts, styles, and image conditions
""" 
import random
import numpy as np
import io
from PIL import Image, ImageEnhance, ImageFilter, ImageFont, ImageDraw, ImageOps
import torchvision.transforms as transforms

class AspectRatioResize:
    """
    Resizes image dynamically keeping aspect ratio, with a fixed height.
    Ensures the width is divisible by 4 to cleanly pass through the CNN.
    """
    def __init__(self, target_height=64, max_width=1024):
        self.target_height = target_height
        self.max_width = max_width

    def __call__(self, img):
        w, h = img.size
        new_w = max(4, int(w * (self.target_height / h)))
        if new_w > self.max_width:
            new_w = self.max_width
            
        # Ensure new_w is divisible by 4 for exact sequence length calculation
        new_w = max(4, (new_w // 4) * 4)
        return img.resize((new_w, self.target_height), Image.BILINEAR)

class OCRAugmentation:
    """
    Custom augmentation for OCR with realistic variations
    """
    def __init__(self, p=0.5):
        """
        Args:
            p: Probability of applying each augmentation
        """
        self.p = p
    
    def __call__(self, img):
        """
        Apply random augmentations to image
        
        Args:
            img: PIL Image
            
        Returns:
            Augmented PIL Image
        """
        # Convert to PIL if tensor
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)
        
        # 1. Random brightness (±30%)
        if random.random() < self.p:
            factor = random.uniform(0.7, 1.3)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(factor)
        
        # 2. Random contrast (±30%)
        if random.random() < self.p:
            factor = random.uniform(0.7, 1.3)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(factor)
        
        # 3. Random sharpness
        if random.random() < self.p:
            factor = random.uniform(0.5, 1.5)
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(factor)
        
        # 4. Slight rotation (±2 degrees for natural skew)
        if random.random() < self.p:
            angle = random.uniform(-2, 2)
            img = img.rotate(angle, fillcolor=255, expand=True)
        
        # 5. Gaussian blur (simulates focus issues)
        if random.random() < self.p * 0.3:  # Less frequent
            radius = random.uniform(0.5, 1.5)
            img = img.filter(ImageFilter.GaussianBlur(radius))
        
        # 6. Add noise (simulates scan artifacts)
        if random.random() < self.p * 0.3:  # Less frequent
            img_array = np.array(img)
            noise = np.random.normal(0, 5, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)
        
        # 7. Random erosion/dilation (simulates bold/thin text)
        if random.random() < self.p * 0.4:
            if random.random() < 0.5:
                # Erosion (thinner text)
                img = img.filter(ImageFilter.MinFilter(3))
            else:
                # Dilation (bolder text)
                img = img.filter(ImageFilter.MaxFilter(3))
        
        # 8. Perspective transform (simulates camera angle)
        if random.random() < self.p * 0.3:
            width, height = img.size
            coeffs = [1 + random.uniform(-0.05, 0.05) for _ in range(8)]
            img = img.transform((width, height), Image.PERSPECTIVE, coeffs)

        # 9. Random padding/cropping (simulates imperfect bounding boxes)
        if random.random() < self.p * 0.4:
            pad_top = random.randint(0, 5)
            pad_bottom = random.randint(0, 5)
            pad_left = random.randint(0, 10)
            pad_right = random.randint(0, 10)
            img = ImageOps.expand(img, (pad_left, pad_top, pad_right, pad_bottom), fill=255)

        # 10. JPEG compression artifacts (simulates real-world images)
        if random.random() < self.p * 0.3:
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=random.randint(40, 85))
            buffer.seek(0)
            img = Image.open(buffer).convert('L')

        # 11. Random line/scratch artifacts
        if random.random() < self.p * 0.1:
            draw = ImageDraw.Draw(img)
            x1 = random.randint(0, img.width)
            y1 = random.randint(0, img.height)
            x2 = random.randint(0, img.width)
            draw.line([(x1, y1), (x2, y1 + random.randint(-3, 3))], fill=128, width=1)

        return img


def get_training_transforms(augment=True, max_width=1024):
    """
    Get training transforms with optional augmentation
    
    Args:
        augment: Whether to apply data augmentation
        max_width: Maximum allowed width for dynamic resizing
        
    Returns:
        torchvision.transforms.Compose
    """
    transform_list = []
    
    if augment:
        # Add custom OCR augmentation first
        transform_list.append(OCRAugmentation(p=0.5))
    
    # Standard transforms (dynamic width)
    transform_list.extend([
        AspectRatioResize(target_height=64, max_width=max_width),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    return transforms.Compose(transform_list)


def get_validation_transforms(max_width=1024):
    """
    Get validation transforms (no augmentation)
    
    Returns:
        torchvision.transforms.Compose
    """
    return transforms.Compose([
        AspectRatioResize(target_height=64, max_width=max_width),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


# Example usage:
if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt
    
    # Load a sample image
    img_path = "data/train_real_sentences/images/sentence_000000.png"
    img = Image.open(img_path)
    
    # Get augmentation
    augmenter = OCRAugmentation(p=0.8)
    
    # Create multiple augmented versions
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    axes = axes.flatten()
    
    # Original
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Augmented versions
    for i in range(1, 8):
        aug_img = augmenter(img.copy())
        axes[i].imshow(aug_img, cmap='gray')
        axes[i].set_title(f'Augmented {i}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_examples.png', dpi=150, bbox_inches='tight')
    print("✓ Saved augmentation examples to: augmentation_examples.png")
    plt.show()

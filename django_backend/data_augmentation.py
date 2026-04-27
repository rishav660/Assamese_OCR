"""
Data Augmentation for OCR Training
Adds variations to make model robust to different fonts, styles, and image conditions
"""
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageFont, ImageDraw
import torchvision.transforms as transforms

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
        
        return img


def get_training_transforms(augment=True):
    """
    Get training transforms with optional augmentation
    
    Args:
        augment: Whether to apply data augmentation
        
    Returns:
        torchvision.transforms.Compose
    """
    transform_list = []
    
    if augment:
        # Add custom OCR augmentation first
        transform_list.append(OCRAugmentation(p=0.5))
    
    # Standard transforms
    transform_list.extend([
        transforms.Resize((32, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    return transforms.Compose(transform_list)


def get_validation_transforms():
    """
    Get validation transforms (no augmentation)
    
    Returns:
        torchvision.transforms.Compose
    """
    return transforms.Compose([
        transforms.Resize((32, 512)),
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

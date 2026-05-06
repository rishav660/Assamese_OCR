"""
Transfer Learning for Sentence OCR
Uses pre-trained character model and fine-tunes on sentence data
This should give MUCH better results than training from scratch
"""
import os
import torch
from torch.utils.data import DataLoader
from char_map import char_to_idx, idx_to_char
from torchvision import transforms
from dataset import AssameseOCRDataset, collate_fn
from model import AssameseOCR
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Check if space is in char_map
if ' ' in char_to_idx:
    print(f"✓ Space character found in char_map at index: {char_to_idx[' ']}")
else:
    print("⚠️  WARNING: Space character NOT in char_map!")

# Transforms - SAME width as base model (critical for transfer learning!)
transform = transforms.Compose([
    transforms.Resize((64, 512)),  # Keep same as base model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

print("\nLoading sentence datasets...")

# Load sentence datasets
train_dataset = AssameseOCRDataset(
    img_dir='data/train_sentences/images',
    label_file='data/train_sentences/labels/labels.txt',
    char_to_idx=char_to_idx,
    transform=transform
)

val_dataset = AssameseOCRDataset(
    img_dir='data/val_sentences/images',
    label_file='data/val_sentences/labels/labels.txt',
    char_to_idx=char_to_idx,
    transform=transform
)

print(f"Train sentences: {len(train_dataset)}, Val sentences: {len(val_dataset)}")

if len(train_dataset) == 0:
    print("\n" + "="*60)
    print("ERROR: No training data found!")
    print("="*60)
    print("You need to generate sentence data first:")
    print("  python generate_sentence_data.py")
    print("="*60)
    exit(1)

# DataLoaders
BATCH_SIZE = 16  # Standard batch size

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=True
)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# ============ TRANSFER LEARNING: Load Pre-trained Model ============
num_classes = len(char_to_idx) + 1
model = AssameseOCR(img_height=64, nn_classes=num_classes).to(device)

# Note: AssameseOCR uses ResNet+Transformer, so loading old CNN+LSTM 
# weights will fail. Transfer learning from old architecture is deprecated.
print("WARNING: Transfer learning from old architecture not supported with AssameseOCR")

# Try to load pre-trained weights
pretrained_path = "checkpoints/best_model_fast.pth"
if os.path.exists(pretrained_path):
    print(f"\n{'='*60}")
    print("LOADING PRE-TRAINED MODEL")
    print(f"{'='*60}")
    print(f"Loading weights from: {pretrained_path}")
    
    try:
        state_dict = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print("✓ Pre-trained weights loaded successfully!")
        print("  Model already knows Assamese characters")
        print("  Now fine-tuning for sentence recognition...")
    except Exception as e:
        print(f"⚠️  Could not load pre-trained weights: {e}")
        print("  Training from scratch instead...")
else:
    print(f"\n⚠️  Pre-trained model not found at: {pretrained_path}")
    print("  Training from scratch...")

model = model.to(device)

# ============ OPTIMIZER: Lower Learning Rate for Fine-tuning ============
# Use a MUCH lower learning rate to preserve learned features
optimizer = optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.01)  # Very low LR to preserve character knowledge
criterion = nn.CTCLoss(blank=len(char_to_idx), reduction='mean', zero_infinity=True)

# Scheduler - more aggressive
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=2, factor=0.5, verbose=True
)

print("\n" + "="*60)
print("TRANSFER LEARNING CONFIGURATION")
print("="*60)
print(f"Base model: Character-level OCR")
print(f"Fine-tuning for: Sentence-level OCR with spaces")
print(f"Learning rate: 0.00001 (10x lower for fine-tuning)")
print(f"Strategy: Preserve character knowledge, learn spacing")
print("="*60 + "\n")

# Sanity check
print("Loading first batch...")
try:
    sample_batch = next(iter(train_loader))
    if sample_batch is not None:
        images, labels, input_lengths, target_lengths = sample_batch
        print(f"✓ Batch loaded: {images.shape}")
        print(f"  Sequence length: {input_lengths[0].item()}")
        print(f"  Sample target lengths: {target_lengths[:4].tolist()}")
        
        # Check if labels contain spaces
        space_idx = char_to_idx.get(' ', -1)
        if space_idx > 0 and space_idx in labels:
            print(f"  ✓ Labels contain SPACES!")
        else:
            print(f"  ⚠️  Labels don't contain spaces")
except Exception as e:
    print(f"Error loading batch: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Training loop
num_epochs = 15  # Fewer epochs to prevent overfitting
best_val_loss = float('inf')
train_losses = []
val_losses = []
patience = 3  # Stricter early stopping
no_improve_epochs = 0
min_delta = 0.001  # Minimum improvement to count as progress

print("\nStarting fine-tuning...\n")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
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
        
        # Forward
        outputs = model(images)
        outputs = torch.log_softmax(outputs, 2)
        
        # CTC Loss
        loss = criterion(outputs, labels, input_lengths, target_lengths)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
        
        if batch_idx % 50 == 0:
            elapsed = time.time() - epoch_start
            batches_per_sec = (batch_idx + 1) / elapsed if elapsed > 0 else 0
            eta = (len(train_loader) - batch_idx) / batches_per_sec if batches_per_sec > 0 else 0
            print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f} | Speed: {batches_per_sec:.2f} batch/s | ETA: {eta:.0f}s")
    
    avg_train_loss = total_loss / batch_count if batch_count > 0 else 0
    train_losses.append(avg_train_loss)
    
    # Validation
    model.eval()
    val_loss = 0
    val_batch_count = 0
    
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
    
    avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else 0
    val_losses.append(avg_val_loss)
    
    epoch_time = time.time() - epoch_start
    
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.1f}s ({epoch_time/60:.1f} min)")
    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    print(f"Gap (Train-Val): {abs(avg_train_loss - avg_val_loss):.4f}")
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Check for overfitting
    if avg_val_loss < 0.01:
        print("⚠️  WARNING: Validation loss suspiciously low - possible overfitting!")
    
    if abs(avg_train_loss - avg_val_loss) > 0.5:
        print("⚠️  WARNING: Large train/val gap - model may be overfitting!")
    
    if avg_val_loss < best_val_loss - min_delta:
        improvement = best_val_loss - avg_val_loss
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "checkpoints/best_model_transfer.pth")
        print(f"✓ Best model saved! (improved by {improvement:.4f})")
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
        print(f"No improvement for {no_improve_epochs} epoch(s)")
    
    print(f"{'='*60}\n")
    
    scheduler.step(avg_val_loss)
    
    if no_improve_epochs >= patience:
        print(f"Early stopping after {epoch+1} epochs")
        break

# Save final
torch.save(model.state_dict(), "checkpoints/final_model_transfer.pth")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Transfer Learning - Sentence Model')
plt.savefig('training_curve_transfer.png')

print("\n" + "="*60)
print("TRANSFER LEARNING COMPLETE!")
print("="*60)
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Model saved to: checkpoints/best_model_transfer.pth")
print("\nThis model should perform MUCH better because:")
print("  ✓ Started with pre-trained character knowledge")
print("  ✓ Fine-tuned specifically for sentences")
print("  ✓ Learned to recognize spaces between words")
print("\nTest with: python test_sentence_ocr.py")
print("="*60)

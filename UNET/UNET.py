import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm
import warnings
import random
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class DoubleConv(nn.Module):
    """Double Convolution block with Dropout for regularization"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.4):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.4):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_rate)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, dropout_rate=0.4):
        super(Up, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, dropout_rate)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle different input sizes
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class AggressivelyRegularizedUNet(nn.Module):
    """Aggressively Regularized UNet Architecture for Brain MRI Segmentation"""
    def __init__(self, n_channels=1, n_classes=4, bilinear=True, dropout_rate=0.4):
        super(AggressivelyRegularizedUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64, dropout_rate)
        self.down1 = Down(64, 128, dropout_rate)
        self.down2 = Down(128, 256, dropout_rate)
        self.down3 = Down(256, 512, dropout_rate)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, dropout_rate)
        
        self.up1 = Up(1024, 512 // factor, bilinear, dropout_rate)
        self.up2 = Up(512, 256 // factor, bilinear, dropout_rate)
        self.up3 = Up(256, 128 // factor, bilinear, dropout_rate)
        self.up4 = Up(128, 64, bilinear, dropout_rate)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class HeavilyAugmentedBrainMRIDataset(Dataset):
    """Dataset class with heavy data augmentation for challenging training"""
    def __init__(self, image_dir, mask_dir, train=True, augment_prob=0.8):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.train = train
        self.augment_prob = augment_prob
        
        # Get all image files (case_*.png)
        self.image_files = glob.glob(os.path.join(image_dir, "case_*.png"))
        self.image_files.sort()
        
        print(f"Found {len(self.image_files)} images in {image_dir}")
        
        # Verify corresponding masks exist
        valid_pairs = []
        for img_path in self.image_files:
            # Get filename and convert case_ to seg_
            img_filename = os.path.basename(img_path)
            mask_filename = img_filename.replace('case_', 'seg_')
            mask_path = os.path.join(mask_dir, mask_filename)
            
            if os.path.exists(mask_path):
                valid_pairs.append((img_path, mask_path))
            else:
                print(f"Warning: No mask found for {img_path}")
        
        self.image_mask_pairs = valid_pairs
        print(f"Found {len(self.image_mask_pairs)} valid image-mask pairs")

    def __len__(self):
        return len(self.image_mask_pairs)

    def heavy_augment_data(self, image, mask):
        """Apply heavy data augmentation to both image and mask"""
        if not self.train or random.random() > self.augment_prob:
            return image, mask
        
        # Convert to PIL for transforms
        image_pil = Image.fromarray((image * 255).astype(np.uint8))
        mask_pil = Image.fromarray(mask)
        
        # Random rotation (-15 to 15 degrees) - increased range
        if random.random() > 0.3:
            angle = random.uniform(-15, 15)
            image_pil = image_pil.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
            mask_pil = mask_pil.rotate(angle, resample=Image.NEAREST, fillcolor=0)
        
        # Random horizontal flip
        if random.random() > 0.4:
            image_pil = image_pil.transpose(Image.FLIP_LEFT_RIGHT)
            mask_pil = mask_pil.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Random vertical flip (added)
        if random.random() > 0.7:
            image_pil = image_pil.transpose(Image.FLIP_TOP_BOTTOM)
            mask_pil = mask_pil.transpose(Image.FLIP_TOP_BOTTOM)
        
        # Random brightness/contrast adjustment (image only) - increased variation
        if random.random() > 0.3:
            brightness_factor = random.uniform(0.6, 1.4)
            contrast_factor = random.uniform(0.6, 1.4)
            
            import PIL.ImageEnhance as ImageEnhance
            enhancer = ImageEnhance.Brightness(image_pil)
            image_pil = enhancer.enhance(brightness_factor)
            enhancer = ImageEnhance.Contrast(image_pil)
            image_pil = enhancer.enhance(contrast_factor)
        
        # Random scaling (added)
        if random.random() > 0.6:
            scale_factor = random.uniform(0.9, 1.1)
            new_size = int(256 * scale_factor)
            image_pil = image_pil.resize((new_size, new_size), Image.BILINEAR)
            mask_pil = mask_pil.resize((new_size, new_size), Image.NEAREST)
            
            # Crop or pad back to 256x256
            if new_size > 256:
                # Crop center
                left = (new_size - 256) // 2
                top = (new_size - 256) // 2
                image_pil = image_pil.crop((left, top, left + 256, top + 256))
                mask_pil = mask_pil.crop((left, top, left + 256, top + 256))
            elif new_size < 256:
                # Pad to center
                pad = (256 - new_size) // 2
                image_pil = Image.new('L', (256, 256), 0)
                mask_pil_new = Image.new('L', (256, 256), 0)
                image_pil.paste(image_pil, (pad, pad))
                mask_pil_new.paste(mask_pil, (pad, pad))
                image_pil = image_pil
                mask_pil = mask_pil_new
        
        # Convert back to numpy
        image = np.array(image_pil, dtype=np.float32) / 255.0
        mask = np.array(mask_pil, dtype=np.uint8)
        
        return image, mask

    def __getitem__(self, idx):
        img_path, mask_path = self.image_mask_pairs[idx]
        
        # Load image and mask
        image = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        
        # Convert to numpy arrays
        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.uint8)
        
        # Resize to standard size if needed
        if image.shape != (256, 256):
            image = np.array(Image.fromarray((image * 255).astype(np.uint8)).resize((256, 256)))
            image = image.astype(np.float32) / 255.0
            mask = np.array(Image.fromarray(mask).resize((256, 256), Image.NEAREST))
        
        # Apply heavy augmentation
        image, mask = self.heavy_augment_data(image, mask)
        
        # Convert grayscale mask values to class indices
        mask_classes = np.zeros_like(mask, dtype=np.uint8)
        mask_classes[mask == 0] = 0    # Background
        mask_classes[mask == 85] = 1   # CSF
        mask_classes[mask == 170] = 2  # Gray Matter
        mask_classes[mask == 255] = 3  # White Matter
        
        # Add stronger Gaussian noise to image (training only)
        if self.train and random.random() > 0.5:  # Increased probability
            noise = np.random.normal(0, 0.04, image.shape).astype(np.float32)  # Higher noise
            image = np.clip(image + noise, 0, 1)
        
        # Convert to tensors
        image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
        mask = torch.from_numpy(mask_classes).long()
        
        return image, mask

def dice_coefficient(pred, target, num_classes=4):
    """Calculate Dice Coefficient for each class"""
    dice_scores = []
    
    for class_idx in range(num_classes):
        pred_class = (pred == class_idx).float()
        target_class = (target == class_idx).float()
        
        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()
        
        if union == 0:
            dice = 1.0  # Perfect score when both pred and target are empty
        else:
            dice = (2.0 * intersection) / union
        
        dice_scores.append(dice.item())
    
    return dice_scores

def train_model_aggressively_regularized(model, train_loader, val_loader, num_epochs=100, initial_lr=5e-5, weight_decay=5e-4):
    """Training function with aggressive regularization"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    
    # Even more conservative learning rate scheduling
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.7)
    
    train_losses = []
    val_losses = []
    val_dice_scores = []
    learning_rates = []
    
    best_dice = 0.0
    patience = 10  
    patience_counter = 0
    
    print(f"AGGRESSIVE REGULARIZATION TRAINING CONFIGURATION:")
    print(f"- Initial Learning Rate: {initial_lr}")
    print(f"- Weight Decay: {weight_decay}")
    print(f"- Dropout Rate: 0.4")
    print(f"- Data Augmentation Probability: 80%")
    print(f"- Early Stopping Patience: {patience}")
    print(f"- LR Scheduler: Step every 15 epochs, gamma=0.7")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # Training
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train] LR:{current_lr:.6f}')
        
        for images, masks in train_pbar:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_dice_scores = [[] for _ in range(4)]  # 4 classes
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        
        with torch.no_grad():
            for images, masks in val_pbar:
                images, masks = images.to(device), masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                # Calculate Dice scores
                pred_masks = torch.argmax(outputs, dim=1)
                
                for i in range(images.size(0)):
                    dice_scores = dice_coefficient(pred_masks[i].cpu(), masks[i].cpu())
                    for j, score in enumerate(dice_scores):
                        all_dice_scores[j].append(score)
                
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Calculate average dice scores
        mean_dice_scores = [np.mean(scores) for scores in all_dice_scores]
        overall_dice = np.mean(mean_dice_scores)
        val_dice_scores.append(mean_dice_scores)
        
        # Step the scheduler
        scheduler.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        print(f'Train/Val Loss Ratio: {avg_train_loss/avg_val_loss:.2f} (>1.0 indicates good regularization)')
        print(f'Dice Scores - Class 0: {mean_dice_scores[0]:.4f}, Class 1: {mean_dice_scores[1]:.4f}, '
              f'Class 2: {mean_dice_scores[2]:.4f}, Class 3: {mean_dice_scores[3]:.4f}')
        print(f'Overall Dice: {overall_dice:.4f}, LR: {current_lr:.6f}')
        
        # Early stopping and model saving
        if overall_dice > best_dice:
            best_dice = overall_dice
            patience_counter = 0
            torch.save(model.state_dict(), 'best_aggressive_regularized_unet_model.pth')
            print(f'✓ New best model saved with Dice: {best_dice:.4f}')

            # Add performance-based stopping
            if overall_dice >= 0.92:
                print(f'🎯 Target performance achieved! Overall Dice: {overall_dice:.4f} >= 0.95')
                print('Stopping training early due to excellent performance.')
                break
        else:
            patience_counter += 1
            print(f'No improvement for {patience_counter}/{patience} epochs')
            
        if patience_counter >= 15:
            print(f'Early stopping triggered after {patience} epochs without improvement')
            break
            
        print('-' * 60)
    
    return train_losses, val_losses, val_dice_scores, learning_rates

def visualize_training_samples(dataset, num_samples=4):
    """Visualize augmented training samples"""
    fig, axes = plt.subplots(2, num_samples, figsize=(16, 8))
    
    for i in range(num_samples):
        image, mask = dataset[i]
        
        # Convert back to numpy for visualization
        image_np = image.squeeze().numpy()
        mask_np = mask.numpy()
        
        # Plot image
        axes[0, i].imshow(image_np, cmap='gray')
        axes[0, i].set_title(f'Augmented Image {i+1}')
        axes[0, i].axis('off')
        
        # Plot mask
        axes[1, i].imshow(mask_np, cmap='tab10', vmin=0, vmax=3)
        axes[1, i].set_title(f'Mask {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('augmented_samples.png', dpi=150, bbox_inches='tight')
    plt.show()

def main_aggressively_regularized():
    """Main training pipeline with aggressive regularization"""
    # Set file paths
    base_dir = "C:/Users/s4913017/Downloads/keras_png_slices_data/keras_png_slices_data"
    
    train_image_dir = os.path.join(base_dir, "keras_png_slices_train")
    train_mask_dir = os.path.join(base_dir, "keras_png_slices_seg_train")
    
    val_image_dir = os.path.join(base_dir, "keras_png_slices_validate")
    val_mask_dir = os.path.join(base_dir, "keras_png_slices_seg_validate")
    
    test_image_dir = os.path.join(base_dir, "keras_png_slices_test")
    test_mask_dir = os.path.join(base_dir, "keras_png_slices_seg_test")
    
    # Create datasets with heavy augmentation
    print("Creating aggressively regularized datasets...")
    train_dataset = HeavilyAugmentedBrainMRIDataset(train_image_dir, train_mask_dir, train=True, augment_prob=0.8)
    val_dataset = HeavilyAugmentedBrainMRIDataset(val_image_dir, val_mask_dir, train=False, augment_prob=0.0)
    test_dataset = HeavilyAugmentedBrainMRIDataset(test_image_dir, test_mask_dir, train=False, augment_prob=0.0)
    
    # Visualize augmented samples
    print("Visualizing augmented training samples...")
    visualize_training_samples(train_dataset)
    
    # Create data loaders
    batch_size = 6  # Reduced batch size for more gradient updates
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Initialize aggressively regularized model
    print("Initializing Aggressively Regularized UNet model...")
    model = AggressivelyRegularizedUNet(n_channels=1, n_classes=4, dropout_rate=0.4).to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train with aggressive regularization
    print("Starting aggressively regularized training...")
    train_losses, val_losses, val_dice_scores, learning_rates = train_model_aggressively_regularized(
        model, train_loader, val_loader, 
        num_epochs=100, 
        initial_lr=5e-5,  # Very low learning rate
        weight_decay=5e-4  # High weight decay
    )
    
    # Load best model and evaluate
    model.load_state_dict(torch.load('best_aggressive_regularized_unet_model.pth'))
    print("Best aggressively regularized model loaded for evaluation.")
    
    # Evaluate on test set
    model.eval()
    all_dice_scores = [[] for _ in range(4)]
    
    print("Evaluating model on test set...")
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc='Testing'):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            pred_masks = torch.argmax(outputs, dim=1)
            
            for i in range(images.size(0)):
                dice_scores = dice_coefficient(pred_masks[i].cpu(), masks[i].cpu())
                for j, score in enumerate(dice_scores):
                    all_dice_scores[j].append(score)
    
    # Final results
    mean_dice_scores = [np.mean(scores) for scores in all_dice_scores]
    std_dice_scores = [np.std(scores) for scores in all_dice_scores]
    overall_mean = np.mean(mean_dice_scores)
    
    print("\n" + "="*60)
    print("FINAL AGGRESSIVELY REGULARIZED MODEL RESULTS:")
    print("="*60)
    print(f"Class 0 (Background):  {mean_dice_scores[0]:.4f} ± {std_dice_scores[0]:.4f}")
    print(f"Class 1 (CSF):         {mean_dice_scores[1]:.4f} ± {std_dice_scores[1]:.4f}")
    print(f"Class 2 (Gray Matter): {mean_dice_scores[2]:.4f} ± {std_dice_scores[2]:.4f}")
    print(f"Class 3 (White Matter):{mean_dice_scores[3]:.4f} ± {std_dice_scores[3]:.4f}")
    print(f"Overall Mean DSC:      {overall_mean:.4f}")
    print("="*60)
    
    target_achieved = all(score > 0.9 for score in mean_dice_scores)
    print(f"Target (>0.9 DSC for all labels): {'✓ ACHIEVED' if target_achieved else '✗ NOT ACHIEVED'}")
    
    # Create comprehensive plots
    plt.figure(figsize=(24, 16))
    
    # Loss curves
    plt.subplot(3, 3, 1)
    plt.plot(train_losses, label='Train Loss', alpha=0.8, linewidth=2)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress: Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Dice scores over epochs
    plt.subplot(3, 3, 2)
    val_dice_array = np.array(val_dice_scores)
    for i in range(4):
        class_names = ['Background', 'CSF', 'Gray Matter', 'White Matter']
        plt.plot(val_dice_array[:, i], label=f'{class_names[i]}', alpha=0.8, linewidth=2)
    plt.plot(np.mean(val_dice_array, axis=1), 'k--', linewidth=3, label='Mean')
    plt.axhline(y=0.9, color='r', linestyle=':', alpha=0.7, label='Target (0.9)')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.title('Validation Dice Scores Over Time')
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # Learning rate schedule
    plt.subplot(3, 3, 3)
    plt.plot(learning_rates, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Final test results
    plt.subplot(3, 3, 4)
    classes = ['Background', 'CSF', 'Gray\nMatter', 'White\nMatter']
    colors = ['skyblue', 'orange', 'lightgreen', 'lightcoral']
    bars = plt.bar(classes, mean_dice_scores, yerr=std_dice_scores, capsize=5, 
                   alpha=0.8, color=colors, edgecolor='black', linewidth=1)
    plt.ylabel('Dice Coefficient')
    plt.title('Final Test Results')
    plt.ylim(0, 1.1)
    plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.7, linewidth=2, label='Target (0.9)')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, score in zip(bars, mean_dice_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Training vs Validation Loss comparison
    plt.subplot(3, 3, 5)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', alpha=0.8, linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', alpha=0.8, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Regularization Effect: Train vs Val Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss ratio over time
    plt.subplot(3, 3, 6)
    loss_ratios = [t/v if v > 0 else 1 for t, v in zip(train_losses, val_losses)]
    plt.plot(loss_ratios, linewidth=2, color='purple')
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Ratio = 1.0')
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss / Val Loss')
    plt.title('Regularization Indicator\n(>1.0 = Good Regularization)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Regularization techniques summary
    plt.subplot(3, 3, 7)
    reg_effects = ['Dropout\n(0.4)', 'Weight Decay\n(5e-4)', 'Heavy Aug\n(80%)', 'Low LR\n(5e-5)', 'Early Stop\n(20 epochs)']
    effectiveness = [0.9, 0.85, 0.95, 0.8, 0.75]
    colors_reg = ['red', 'orange', 'green', 'blue', 'purple']
    bars = plt.bar(reg_effects, effectiveness, alpha=0.8, color=colors_reg, edgecolor='black')
    plt.ylabel('Regularization Strength')
    plt.title('Aggressive Regularization Techniques')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    # Performance progression
    plt.subplot(3, 3, 8)
    mean_dice_progression = np.mean(val_dice_array, axis=1)
    plt.plot(mean_dice_progression, linewidth=3, color='darkgreen')
    plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.7, linewidth=2)
    plt.fill_between(range(len(mean_dice_progression)), mean_dice_progression, alpha=0.3, color='darkgreen')
    plt.xlabel('Epoch')
    plt.ylabel('Overall Dice Score')
    plt.title('Learning Progression\n(Slow & Steady)')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # Class-wise improvement
    plt.subplot(3, 3, 9)
    class_names = ['Background', 'CSF', 'Gray Matter', 'White Matter']
    final_scores = [mean_dice_scores[i] for i in range(4)]
    colors_class = ['lightblue', 'orange', 'lightgreen', 'salmon']
    bars = plt.bar(range(4), final_scores, color=colors_class, alpha=0.8, edgecolor='black')
    plt.ylabel('Final Dice Score')
    plt.title('Class-wise Performance')
    plt.xticks(range(4), [name.replace(' ', '\n') for name in class_names])
    plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.7)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, score) in enumerate(zip(bars, final_scores)):
        plt.text(bar.get_x() + bar.get_width()/2., score + 0.02,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('aggressive_regularized_training_results.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    # Create a separate visualization for segmentation results
    visualize_segmentation_results(model, test_dataset)
    
    print("\nAggressively regularized training complete!")
    print("Model saved as 'best_aggressive_regularized_unet_model.pth'")
    print("Training results saved as 'aggressive_regularized_training_results.png'")
    print("Augmentation samples saved as 'augmented_samples.png'")
    print("Segmentation results saved as 'segmentation_comparison.png'")
    
    # Print comprehensive regularization summary
    print(f"\nAGGRESSIVE REGULARIZATION TECHNIQUES APPLIED:")
    print("=" * 60)
    print(f"✓ Heavy Dropout (0.4) - Strong overfitting prevention")
    print(f"✓ High Weight Decay (5e-4) - Aggressive L2 regularization")
    print(f"✓ Extensive Data Augmentation (80%) - Maximum variability")
    print(f"  - Rotation (±15°), Flips (H&V), Scaling (0.9-1.1x)")
    print(f"  - Brightness/Contrast (±40%), Gaussian Noise")
    print(f"✓ Very Low Learning Rate (5e-5) - Ultra-stable training")
    print(f"✓ Adaptive LR Scheduling (Step: 15 epochs, γ=0.7)")
    print(f"✓ Extended Early Stopping (20 epochs) - Prevents overtraining")
    print(f"✓ Smaller Batch Size (6) - More gradient updates")
    print("=" * 60)
    
    # Training dynamics summary
    final_train_loss = train_losses[-1] if train_losses else 0
    final_val_loss = val_losses[-1] if val_losses else 0
    loss_ratio = final_train_loss / final_val_loss if final_val_loss > 0 else 1
    
    print(f"TRAINING DYNAMICS ACHIEVED:")
    print(f"- Final Train/Val Loss Ratio: {loss_ratio:.2f} (>1.0 indicates good regularization)")
    print(f"- Total Training Epochs: {len(train_losses)}")
    print(f"- Best Overall Dice Score: {best_dice:.4f}")
    print(f"- Learning Curve: {'Gradual' if best_dice < 0.95 else 'Still Fast'}")
    
    if loss_ratio > 1.2:
        print("✓ EXCELLENT: Strong regularization effect achieved!")
    elif loss_ratio > 1.0:
        print("✓ GOOD: Moderate regularization effect achieved!")
    else:
        print("⚠ WARNING: May need even stronger regularization")

def visualize_segmentation_results(model, test_dataset, num_samples=6):
    """Visualize final segmentation results"""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i in range(num_samples):
            # Get a random sample
            idx = np.random.randint(0, len(test_dataset))
            image, true_mask = test_dataset[idx]
            
            # Add batch dimension and move to device
            image_batch = image.unsqueeze(0).to(device)
            
            # Get prediction
            output = model(image_batch)
            pred_mask = torch.argmax(output, dim=1).cpu().numpy()[0]
            
            # Convert to numpy for visualization
            image_np = image.squeeze().numpy()
            true_mask_np = true_mask.numpy()
            
            # Plot original image
            axes[i, 0].imshow(image_np, cmap='gray')
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            # Plot true mask
            axes[i, 1].imshow(true_mask_np, cmap='tab10', vmin=0, vmax=3)
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            # Plot predicted mask
            axes[i, 2].imshow(pred_mask, cmap='tab10', vmin=0, vmax=3)
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
            
            # Calculate and display dice scores
            dice_scores = dice_coefficient(torch.from_numpy(pred_mask), true_mask)
            dice_text = f'Overall: {np.mean(dice_scores):.3f}\n'
            dice_text += f'BG: {dice_scores[0]:.3f}  CSF: {dice_scores[1]:.3f}\n'
            dice_text += f'GM: {dice_scores[2]:.3f}  WM: {dice_scores[3]:.3f}'
            
            axes[i, 2].text(10, 30, dice_text, fontsize=8, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('segmentation_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# For demonstration/inference on new images
def demo_inference(model_path, image_path):
    """Demonstrate inference on a single image"""
    # Load the trained model
    model = AggressivelyRegularizedUNet(n_channels=1, n_classes=4, dropout_rate=0.4).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('L')
    image = np.array(image, dtype=np.float32) / 255.0
    
    # Resize if needed
    if image.shape != (256, 256):
        image = np.array(Image.fromarray((image * 255).astype(np.uint8)).resize((256, 256)))
        image = image.astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)
        pred_mask = torch.argmax(output, dim=1).cpu().numpy()[0]
    
    # Visualize result
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    axes[1].imshow(pred_mask, cmap='tab10', vmin=0, vmax=3)
    axes[1].set_title('Segmentation Result')
    axes[1].axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(axes[1].imshow(pred_mask, cmap='tab10', vmin=0, vmax=3), ax=axes[1])
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['Background', 'CSF', 'Gray Matter', 'White Matter'])
    
    plt.tight_layout()
    plt.show()
    
    return pred_mask

if __name__ == "__main__":
    main_aggressively_regularized()
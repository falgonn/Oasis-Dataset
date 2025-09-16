import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import umap

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class MRIDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.images = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.enc_conv1 = nn.Conv2d(1, 32, 4, 2, 1)   # 64x64 -> 32x32
        self.enc_conv2 = nn.Conv2d(32, 64, 4, 2, 1)  # 32x32 -> 16x16
        self.enc_conv3 = nn.Conv2d(64, 128, 4, 2, 1) # 16x16 -> 8x8
        self.enc_conv4 = nn.Conv2d(128, 256, 4, 2, 1)# 8x8 -> 4x4

        self.fc_mu = nn.Linear(256*4*4, latent_dim)
        self.fc_logvar = nn.Linear(256*4*4, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 256*4*4)
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.dec_conv4 = nn.ConvTranspose2d(32, 1, 4, 2, 1)

    def encode(self, x):
        x = F.leaky_relu(self.enc_conv1(x), 0.2)
        x = F.leaky_relu(self.enc_conv2(x), 0.2)
        x = F.leaky_relu(self.enc_conv3(x), 0.2)
        x = F.leaky_relu(self.enc_conv4(x), 0.2)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = torch.clamp(self.fc_logvar(x), -20, 20)  # Prevent extreme values
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = F.leaky_relu(self.fc_dec(z), 0.2)
        x = x.view(-1, 256, 4, 4)
        x = F.leaky_relu(self.dec_conv1(x), 0.2)
        x = F.leaky_relu(self.dec_conv2(x), 0.2)
        x = F.leaky_relu(self.dec_conv3(x), 0.2)
        x = torch.sigmoid(self.dec_conv4(x))
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """Standard VAE loss function"""
    batch_size = x.size(0)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / batch_size
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

def train_vae(train_dir, val_dir, test_dir, img_size=64, batch_size=32, latent_dim=128,
              epochs=50, lr=1e-4, device='cuda'):

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    train_dataset = MRIDataset(train_dir, transform=transform)
    val_dataset = MRIDataset(val_dir, transform=transform)
    test_dataset = MRIDataset(test_dir, transform=transform)

    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = VAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10

    for epoch in range(epochs):
        model.train()
        total_loss, total_recon, total_kl = 0, 0, 0

        for imgs in train_loader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            recon_imgs, mu, logvar = model(imgs)
            loss, recon_loss, kl_loss = vae_loss(recon_imgs, imgs, mu, logvar, beta=1.0)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

        # Validation
        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for imgs in val_loader:
                imgs = imgs.to(device)
                recon_imgs, mu, logvar = model(imgs)
                loss, _, _ = vae_loss(recon_imgs, imgs, mu, logvar, beta=1.0)
                val_loss_total += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss_total / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} (Recon: {total_recon/len(train_loader):.4f}, KL: {total_kl/len(train_loader):.4f}) | "
              f"Val Loss: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_vae_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return model, train_losses, val_losses, train_loader, val_loader, test_loader

def visualize_reconstructions(model, data_loader, device, num_samples=8):
    """Show original vs reconstructed images"""
    model.eval()
    with torch.no_grad():
        imgs = next(iter(data_loader))[:num_samples].to(device)
        recon_imgs, _, _ = model(imgs)
        
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 4))
        for i in range(num_samples):
            # Original
            axes[0, i].imshow(imgs[i].cpu().squeeze(), cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # Reconstructed
            axes[1, i].imshow(recon_imgs[i].cpu().squeeze(), cmap='gray')
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig('reconstructions.png', dpi=300, bbox_inches='tight')
        plt.show()

def generate_samples(model, device, num_samples=16):
    """Generate new brain images by sampling from latent space"""
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim).to(device)
        samples = model.decode(z)
        
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i in range(num_samples):
            row, col = i // 4, i % 4
            axes[row, col].imshow(samples[i].cpu().squeeze(), cmap='gray')
            axes[row, col].axis('off')
        
        plt.suptitle('Generated Brain MRI Samples')
        plt.tight_layout()
        plt.savefig('generated_samples.png', dpi=300, bbox_inches='tight')
        plt.show()

def visualize_latent_space_umap(model, data_loader, device, num_samples=2000):
    """Visualize the latent space using UMAP"""
    model.eval()
    latent_vectors = []
    
    with torch.no_grad():
        count = 0
        for imgs in data_loader:
            if count >= num_samples:
                break
            
            imgs = imgs.to(device)
            mu, _ = model.encode(imgs)
            latent_vectors.append(mu.cpu().numpy())
            count += len(imgs)
    
    latent_vectors = np.concatenate(latent_vectors, axis=0)[:num_samples]
    
    print("Applying UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(latent_vectors)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=range(len(embedding)), 
                         cmap='viridis', alpha=0.6, s=20)
    plt.colorbar(scatter)
    plt.title('UMAP Visualization of VAE Latent Space')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.grid(True, alpha=0.3)
    plt.savefig('latent_space_umap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return embedding

def interpolate_latent_space(model, data_loader, device, num_steps=10):
    """Show interpolation between two random points in latent space"""
    model.eval()
    
    with torch.no_grad():
        # Get two random latent vectors
        imgs = next(iter(data_loader))[:2].to(device)
        mu1, _ = model.encode(imgs[:1])
        mu2, _ = model.encode(imgs[1:2])
        
        # Interpolate between them
        fig, axes = plt.subplots(1, num_steps, figsize=(15, 3))
        
        for i, alpha in enumerate(np.linspace(0, 1, num_steps)):
            z_interp = alpha * mu1 + (1 - alpha) * mu2
            img_interp = model.decode(z_interp)
            
            axes[i].imshow(img_interp.cpu().squeeze(), cmap='gray')
            axes[i].set_title(f'α={alpha:.1f}')
            axes[i].axis('off')
        
        plt.suptitle('Latent Space Interpolation')
        plt.tight_layout()
        plt.savefig('latent_interpolation.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data paths
    train_dir = "C:/Users/s4913017/Downloads/keras_png_slices_data/keras_png_slices_data/keras_png_slices_train"
    val_dir   = "C:/Users/s4913017/Downloads/keras_png_slices_data/keras_png_slices_data/keras_png_slices_validate"
    test_dir  = "C:/Users/s4913017/Downloads/keras_png_slices_data/keras_png_slices_data/keras_png_slices_test"

    # Train the VAE
    print("Training Standard VAE on Brain MRI Images...")
    model, train_losses, val_losses, train_loader, val_loader, test_loader = train_vae(
        train_dir, val_dir, test_dir,
        img_size=64,
        batch_size=32,
        latent_dim=128,
        epochs=50,
        lr=1e-4,
        device=device
    )

    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('VAE Training Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save the final model
    torch.save(model.state_dict(), 'vae_brain_mri_final.pth')
    print("Model saved as 'vae_brain_mri_final.pth'")
    
    print("\nGenerating visualizations...")
    
    # 1. Show reconstructions
    print("1. Creating reconstruction visualizations...")
    visualize_reconstructions(model, test_loader, device)
    
    # 2. Generate new samples
    print("2. Generating new samples...")
    generate_samples(model, device)
    
    # 3. UMAP visualization of latent space
    print("3. Creating UMAP visualization...")
    embedding = visualize_latent_space_umap(model, test_loader, device)
    
    # 4. Latent space interpolation
    print("4. Creating latent space interpolation...")
    interpolate_latent_space(model, test_loader, device)
    
    print("\nAll visualizations complete!")
    print("Generated files:")
    print("- training_curves.png")
    print("- reconstructions.png")
    print("- generated_samples.png")
    print("- latent_space_umap.png")
    print("- latent_interpolation.png")
    print("- best_vae_model.pth")
    print("- vae_brain_mri_final.pth")

if __name__ == "__main__":
    main()
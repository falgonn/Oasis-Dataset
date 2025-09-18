import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
import math

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def get_data_loaders(batch_size=128):

    """
    import torch
    import torchvision
    import torchvision.transforms as transforms

    # Load CIFAR-10 without normalization
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                       download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False)

    # Calculate mean and std
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in dataloader:
        # data shape: [batch_size, channels, height, width]
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2)**0.5

    print(f"Mean: {mean}")  # this will give [0.4914, 0.4822, 0.4465]
    print(f"Std: {std}")    # this will give [0.2023, 0.1994, 0.2010]
    """
    
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=1, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=1, pin_memory=True
    )

    return train_loader, test_loader

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_planes = 64

        # Your original architecture was correct
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def linear_onecycle_schedule(step, total_steps, peak_value=0.1, pct_start=0.3, 
                           pct_final=0.85, div_factor=25., final_div_factor=1e4):
  
    initial_lr = peak_value / div_factor
    final_lr = initial_lr / final_div_factor
    
    phase_1_steps = int(total_steps * pct_start)
    phase_2_steps = int(total_steps * pct_final) - phase_1_steps
    phase_3_steps = total_steps - phase_1_steps - phase_2_steps
    
    if step < phase_1_steps:
        # Phase 1: linear increase to peak
        lr = initial_lr + (peak_value - initial_lr) * (step / phase_1_steps)
    elif step < phase_1_steps + phase_2_steps:
        # Phase 2: linear decrease from peak
        progress = (step - phase_1_steps) / phase_2_steps
        lr = peak_value - (peak_value - initial_lr) * progress
    else:
        # Phase 3: linear decrease to final
        progress = (step - phase_1_steps - phase_2_steps) / phase_3_steps
        lr = initial_lr - (initial_lr - final_lr) * progress
    
    return max(lr, final_lr)

def train_fixed_cifar10(epochs=30, batch_size=128):
    
    print("Setting up data loaders...")
    train_loader, test_loader = get_data_loaders(batch_size)

    print("Initializing model...")
    model = ResNet18(num_classes=10).to(device)

    # Mixed precision
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    total_steps = len(train_loader) * epochs
    
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    
    # Standard cross entropy (no label smoothing initially)
    criterion = nn.CrossEntropyLoss()

    # Training tracking
    train_losses = []
    test_accuracies = []

    print(f"Starting training for {epochs} epochs...")
    print(f"Total steps: {total_steps}")
    start_time = time.time()

    global_step = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        epoch_start = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

            # CRITICAL: Update learning rate every step, not every epoch
            current_lr = linear_onecycle_schedule(
                global_step, total_steps, 
                peak_value=0.12,
                pct_start=12./epochs,  
                pct_final=25./epochs,  
                div_factor=20.,       
                final_div_factor=200. 
            )
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    output = model(data)
                    loss = criterion(output, target)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            global_step += 1

            # Print progress every 100 steps
            if global_step % 100 == 0:
                train_accuracy = 100. * correct / total
                print(f'[Step {global_step}, Loss {loss:.5f}] Train accuracy: {train_accuracy:.3f}%, LR: {current_lr:.6f}')

        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        train_losses.append(epoch_loss)

      
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        output = model(data)
                else:
                    output = model(data)

                _, predicted = torch.max(output.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()

        test_acc = 100. * test_correct / test_total
        test_accuracies.append(test_acc)

        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time

        print(f'Epoch [{epoch+1}/{epochs}] completed in {epoch_time:.2f}s')
        print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        print(f'Learning Rate: {current_lr:.6f}')
        print(f'Total Time: {total_time:.2f}s')
        print('-' * 50)

        # Early stopping if we achieve target accuracy
        if test_acc >= 93.0:
            print(f"TARGET ACHIEVED! Test accuracy: {test_acc:.2f}%")
            print(f"Training completed in {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
            break

    total_training_time = time.time() - start_time
    final_accuracy = test_accuracies[-1]

    print(f"\nTraining completed!")
    print(f"Total training time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
    print(f"Final test accuracy: {final_accuracy:.2f}%")
    print(f"Target achieved: {'YES' if final_accuracy >= 93.0 else 'NO'}")

    return model, final_accuracy, total_training_time

def main():
    """
    Fixed main function
    """
    print("=" * 60)
    print("CIFAR-10 DAWNBench Challenge")
    print("=" * 60)

    model, final_accuracy, training_time = train_fixed_cifar10(epochs=30, batch_size=128)

    # Quick test
    _, test_loader = get_data_loaders(batch_size=128)
    
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if batch_idx >= 10:  # Just test 10 batches
                break
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    output = model(data)
            else:
                output = model(data)

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    inference_time = time.time() - start_time
    inference_accuracy = 100. * correct / total

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Final accuracy: {final_accuracy:.2f}%")
    print(f"Target achieved: {'YES' if final_accuracy >= 93.0 else 'NO'}")
    print(f"Inference time (10 batches): {inference_time:.4f} seconds")
    print(f"Inference accuracy: {inference_accuracy:.2f}%")

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'accuracy': final_accuracy,
        'training_time': training_time
    }, 'fixed_cifar10_model.pth')
    print("\nModel saved as 'fixed_cifar10_model.pth'")

    return model, final_accuracy, training_time

if __name__ == "__main__":
    model, accuracy, time_taken = main()

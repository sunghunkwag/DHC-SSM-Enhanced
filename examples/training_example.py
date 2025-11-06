"""
Training example using real CIFAR-10 dataset.

This replaces the previous dummy data approach with actual validation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dhc_ssm.core.model import DHCSSMModel, DHCSSMConfig


def get_cifar10_loaders(batch_size=32):
    """Load CIFAR-10 dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    trainloader = DataLoader(
        trainset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=2
    )
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True, 
        transform=transform
    )
    testloader = DataLoader(
        testset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=2
    )
    
    return trainloader, testloader


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        metrics = model.train_step((data, target), optimizer)
        
        total_loss += metrics['loss']
        total_acc += metrics['accuracy']
        num_batches += 1
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(loader)}, "
                  f"Loss: {metrics['loss']:.4f}, "
                  f"Acc: {metrics['accuracy']:.4f}")
    
    return {
        'loss': total_loss / num_batches,
        'accuracy': total_acc / num_batches
    }


def evaluate(model, loader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            
            metrics = model.evaluate_step((data, target))
            
            total_loss += metrics['loss']
            total_acc += metrics['accuracy']
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'accuracy': total_acc / num_batches
    }


def main():
    """Main training loop."""
    # Configuration
    config = DHCSSMConfig(
        input_channels=3,
        hidden_dim=64,
        state_dim=64,
        output_dim=10
    )
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model
    model = DHCSSMModel(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Data
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders(batch_size=32)
    
    # Training
    num_epochs = 10
    print(f"\nTraining for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        test_metrics = evaluate(model, test_loader, device)
        
        print(f"Train Loss: {train_metrics['loss']:.4f}, "
              f"Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"Test Loss: {test_metrics['loss']:.4f}, "
              f"Test Acc: {test_metrics['accuracy']:.4f}")
    
    print("\nTraining complete!")
    print(f"Final Test Accuracy: {test_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
"""
Training Example for DHC-SSM v3.0

This example demonstrates how to train the DHC-SSM model.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dhc_ssm import DHCSSMModel, Trainer, get_debug_config
from dhc_ssm.core.learning_engine import DeterministicOptimizer


def create_dummy_dataset(num_samples=200, num_classes=10):
    """Create a dummy dataset for demonstration."""
    # Random images
    images = torch.randn(num_samples, 3, 32, 32)
    # Random labels
    labels = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(images, labels)


def main():
    print("=" * 70)
    print("DHC-SSM v3.0 - Training Example")
    print("=" * 70)
    
    # Configuration
    print("\n1. Setting up configuration...")
    config = get_debug_config()  # Use debug config for fast training
    config.update(
        num_epochs=5,
        batch_size=8,
        learning_rate=1e-3,
        log_interval=5,
    )
    print(f"   ✓ Configuration: {config.spatial_dim}D spatial, {config.temporal_dim}D temporal")
    
    # Create model
    print("\n2. Creating model...")
    model = DHCSSMModel(config)
    print(f"   ✓ Model created with {model.num_parameters:,} parameters")
    
    # Create datasets
    print("\n3. Creating datasets...")
    train_dataset = create_dummy_dataset(num_samples=160, num_classes=config.output_dim)
    val_dataset = create_dummy_dataset(num_samples=40, num_classes=config.output_dim)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for debugging
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    print(f"   ✓ Train samples: {len(train_dataset)}")
    print(f"   ✓ Val samples: {len(val_dataset)}")
    print(f"   ✓ Batch size: {config.batch_size}")
    
    # Create optimizer
    print("\n4. Creating optimizer...")
    optimizer = DeterministicOptimizer(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        gradient_clip=config.gradient_clip,
    )
    print(f"   ✓ Optimizer: AdamW (lr={config.learning_rate})")
    
    # Create trainer
    print("\n5. Creating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        checkpoint_dir="./checkpoints",
        log_dir="./logs",
    )
    print(f"   ✓ Trainer initialized")
    
    # Test single training step
    print("\n6. Testing single training step...")
    sample_batch = next(iter(train_loader))
    sample_x, sample_y = sample_batch
    
    metrics = model.train_step(sample_x, sample_y, optimizer)
    print(f"   ✓ Training step successful")
    print(f"   - Loss: {metrics.get('total', 0):.4f}")
    print(f"   - Metrics: {list(metrics.keys())}")
    
    # Test single evaluation step
    print("\n7. Testing single evaluation step...")
    val_batch = next(iter(val_loader))
    val_x, val_y = val_batch
    
    val_metrics = model.evaluate_step(val_x, val_y)
    print(f"   ✓ Evaluation step successful")
    print(f"   - Loss: {val_metrics.get('total', 0):.4f}")
    print(f"   - Accuracy: {val_metrics.get('accuracy', 0):.4f}")
    
    # Run training
    print("\n8. Starting training...")
    print(f"   Training for {config.num_epochs} epochs...")
    print("-" * 70)
    
    history = trainer.train(
        num_epochs=config.num_epochs,
        early_stopping_patience=10,
    )
    
    print("-" * 70)
    print("\n9. Training completed!")
    
    # Print training summary
    if history['train']:
        final_train = history['train'][-1]
        print(f"\n   Final Training Metrics:")
        for key, stats in final_train.items():
            if isinstance(stats, dict) and 'mean' in stats:
                print(f"     - {key}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    if history['val']:
        final_val = history['val'][-1]
        print(f"\n   Final Validation Metrics:")
        for key, stats in final_val.items():
            if isinstance(stats, dict) and 'mean' in stats:
                print(f"     - {key}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    # Test final model
    print("\n10. Testing final model...")
    model.eval()
    with torch.no_grad():
        test_x, test_y = next(iter(val_loader))
        test_pred = model(test_x)
        test_acc = (test_pred.argmax(dim=-1) == test_y).float().mean()
    
    print(f"   ✓ Test accuracy: {test_acc:.4f}")
    
    print("\n" + "=" * 70)
    print("✓ Training example completed successfully!")
    print("=" * 70)
    print(f"\nCheckpoints saved to: ./checkpoints")
    print(f"Logs saved to: ./logs")
    print(f"\nTo view training logs, run:")
    print(f"  tensorboard --logdir=./logs")


if __name__ == "__main__":
    main()

"""
Basic Usage Example for DHC-SSM v3.0

This example demonstrates the basic usage of the DHC-SSM architecture.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dhc_ssm import DHCSSMModel, get_default_config, get_small_config


def main():
    print("=" * 70)
    print("DHC-SSM v3.0 - Basic Usage Example")
    print("=" * 70)
    
    # Create model with small configuration for quick testing
    print("\n1. Creating model with small configuration...")
    config = get_small_config()
    model = DHCSSMModel(config)
    
    print(f"   ✓ Model created successfully")
    print(f"   - Parameters: {model.num_parameters:,}")
    print(f"   - Device: {model.device}")
    
    # Create sample input
    print("\n2. Creating sample input...")
    batch_size = 4
    channels = 3
    height = 32
    width = 32
    
    x = torch.randn(batch_size, channels, height, width)
    print(f"   ✓ Input shape: {x.shape}")
    
    # Forward pass
    print("\n3. Running forward pass...")
    model.eval()
    with torch.no_grad():
        predictions = model(x)
    
    print(f"   ✓ Forward pass successful")
    print(f"   - Output shape: {predictions.shape}")
    print(f"   - Output range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    
    # Get predictions
    pred_classes = predictions.argmax(dim=-1)
    print(f"   - Predicted classes: {pred_classes.tolist()}")
    
    # Get model diagnostics
    print("\n4. Model diagnostics...")
    diagnostics = model.get_diagnostics()
    
    print(f"   Architecture: {diagnostics['architecture']}")
    print(f"   Complexity: {diagnostics['complexity']}")
    print(f"   \n   Layer Information:")
    for layer_name, layer_info in diagnostics['layers'].items():
        print(f"     - {layer_name}:")
        print(f"       Type: {layer_info['type']}")
        print(f"       Output dim: {layer_info['output_dim']}")
        print(f"       Complexity: {layer_info.get('complexity', 'N/A')}")
    
    # Forward pass with features
    print("\n5. Running forward pass with intermediate features...")
    with torch.no_grad():
        predictions, features = model(x, return_features=True)
    
    print(f"   ✓ Forward pass with features successful")
    print(f"   - Available features: {list(features.keys())}")
    for feat_name, feat_tensor in features.items():
        print(f"     {feat_name}: {feat_tensor.shape}")
    
    # Test with different batch sizes
    print("\n6. Testing with different batch sizes...")
    for bs in [1, 2, 8]:
        x_test = torch.randn(bs, channels, height, width)
        with torch.no_grad():
            pred_test = model(x_test)
        print(f"   ✓ Batch size {bs}: input {x_test.shape} → output {pred_test.shape}")
    
    print("\n" + "=" * 70)
    print("✓ All tests passed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

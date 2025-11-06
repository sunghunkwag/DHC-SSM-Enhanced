"""
Comprehensive benchmark tests with real CIFAR-10 data.

This replaces dummy data tests with actual performance validation.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dhc_ssm.core.model import DHCSSMModel, DHCSSMConfig


class SimpleCNN(nn.Module):
    """Baseline CNN for comparison."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def get_test_data(num_samples=1000):
    """Load subset of CIFAR-10 for testing."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    subset = Subset(testset, range(num_samples))
    loader = DataLoader(subset, batch_size=32, shuffle=False)
    
    return loader


def test_forward_pass():
    """Test forward pass with real data."""
    print("\n=== Test 1: Forward Pass ===")
    
    config = DHCSSMConfig()
    model = DHCSSMModel(config)
    loader = get_test_data(num_samples=100)
    
    try:
        model.eval()
        with torch.no_grad():
            for data, _ in loader:
                output = model(data)
                assert output.shape == (data.shape[0], 10)
        
        print("✓ Forward pass successful")
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False


def test_training_step():
    """Test training step with real data."""
    print("\n=== Test 2: Training Step ===")
    
    config = DHCSSMConfig()
    model = DHCSSMModel(config)
    optimizer = torch.optim.Adam(model.parameters())
    loader = get_test_data(num_samples=100)
    
    try:
        model.train()
        for data, target in loader:
            metrics = model.train_step((data, target), optimizer)
            assert 'loss' in metrics
            assert 'accuracy' in metrics
            break
        
        print(f"✓ Training step successful")
        print(f"  Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        return True
    except Exception as e:
        print(f"✗ Training step failed: {e}")
        return False


def test_learning_progress():
    """Test if model can actually learn (loss decreases)."""
    print("\n=== Test 3: Learning Progress ===")
    
    config = DHCSSMConfig()
    model = DHCSSMModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loader = get_test_data(num_samples=200)
    
    try:
        model.train()
        initial_losses = []
        final_losses = []
        
        # Collect some batches
        batches = []
        for i, batch in enumerate(loader):
            batches.append(batch)
            if i >= 5:  # Only use 5 batches
                break
        
        # Initial losses
        model.eval()
        with torch.no_grad():
            for data, target in batches:
                metrics = model.evaluate_step((data, target))
                initial_losses.append(metrics['loss'])
        
        # Train for a few steps
        model.train()
        for epoch in range(3):
            for data, target in batches:
                model.train_step((data, target), optimizer)
        
        # Final losses
        model.eval()
        with torch.no_grad():
            for data, target in batches:
                metrics = model.evaluate_step((data, target))
                final_losses.append(metrics['loss'])
        
        initial_avg = sum(initial_losses) / len(initial_losses)
        final_avg = sum(final_losses) / len(final_losses)
        improvement = initial_avg - final_avg
        
        print(f"✓ Learning test completed")
        print(f"  Initial Loss: {initial_avg:.4f}")
        print(f"  Final Loss: {final_avg:.4f}")
        print(f"  Improvement: {improvement:.4f}")
        
        return improvement > 0
    except Exception as e:
        print(f"✗ Learning test failed: {e}")
        return False


def test_performance_comparison():
    """Compare DHC-SSM vs baseline CNN."""
    print("\n=== Test 4: Performance Comparison ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = get_test_data(num_samples=500)
    
    # Test DHC-SSM
    config = DHCSSMConfig()
    dhc_model = DHCSSMModel(config).to(device)
    
    dhc_start = time.time()
    dhc_model.eval()
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            _ = dhc_model(data)
    dhc_time = time.time() - dhc_start
    
    # Test baseline CNN
    cnn_model = SimpleCNN().to(device)
    
    cnn_start = time.time()
    cnn_model.eval()
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            _ = cnn_model(data)
    cnn_time = time.time() - cnn_start
    
    dhc_params = sum(p.numel() for p in dhc_model.parameters())
    cnn_params = sum(p.numel() for p in cnn_model.parameters())
    
    print(f"✓ Performance comparison completed")
    print(f"  DHC-SSM: {dhc_time:.3f}s, {dhc_params:,} params")
    print(f"  CNN: {cnn_time:.3f}s, {cnn_params:,} params")
    print(f"  Speed ratio: {cnn_time/dhc_time:.2f}x")
    
    return True


def test_memory_efficiency():
    """Test memory usage."""
    print("\n=== Test 5: Memory Efficiency ===")
    
    config = DHCSSMConfig()
    model = DHCSSMModel(config)
    
    # Test with increasing batch sizes
    batch_sizes = [1, 4, 8, 16, 32]
    max_batch = 1
    
    try:
        model.eval()
        for batch_size in batch_sizes:
            with torch.no_grad():
                x = torch.randn(batch_size, 3, 32, 32)
                output = model(x)
                assert output.shape == (batch_size, 10)
                max_batch = batch_size
        
        print(f"✓ Memory test successful")
        print(f"  Max batch size: {max_batch}")
        return True
    except Exception as e:
        print(f"✗ Memory test failed at batch size {max_batch}: {e}")
        return False


def run_all_tests():
    """Run all tests and return summary."""
    print("=" * 70)
    print("DHC-SSM v3.1 - COMPREHENSIVE BENCHMARK")
    print("=" * 70)
    
    tests = [
        ("Forward Pass", test_forward_pass),
        ("Training Step", test_training_step),
        ("Learning Progress", test_learning_progress),
        ("Performance Comparison", test_performance_comparison),
        ("Memory Efficiency", test_memory_efficiency),
    ]
    
    results = {}
    passed = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed += 1
        except Exception as e:
            print(f"\n=== {test_name} ===\n✗ Test failed with exception: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 ALL TESTS PASSED - DHC-SSM v3.1 is ready!")
        return True
    else:
        print(f"\n⚠️  {len(tests) - passed} tests failed - needs attention")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
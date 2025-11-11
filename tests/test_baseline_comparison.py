"""
Baseline Comparison Tests for DHC-SSM vs Standard Architectures.

This module compares DHC-SSM against standard baselines (MLP, CNN) for both
inference speed and model capacity to provide objective performance metrics.
"""

import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import time
import sys
import os
import json
from typing import Dict, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dhc_ssm.core.model import DHCSSMConfig
from dhc_ssm.adapters.rl_policy import RLPolicyAdapter


class MLPPolicy(nn.Module):
    """Simple MLP baseline for RL tasks."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.network(x)


class CNNPolicy(nn.Module):
    """Simple CNN baseline for RL tasks (for comparison)."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        # Reshape observation to square
        self.obs_dim = obs_dim
        self.reshape_dim = max(4, int(np.ceil(np.sqrt(obs_dim))))
        
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, self.reshape_dim ** 2),
            nn.ReLU()
        )
        
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        features = self.feature_net(x)
        features = features.view(batch_size, 1, self.reshape_dim, self.reshape_dim)
        conv_out = self.conv_net(features).squeeze(-1).squeeze(-1)
        return self.action_head(conv_out)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_speed(model: nn.Module, input_tensor: torch.Tensor, 
                           num_iterations: int = 1000, warmup: int = 10) -> Dict:
    """Measure inference speed with warmup."""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    
    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(input_tensor)
    elapsed_time = time.time() - start_time
    
    avg_time = elapsed_time / num_iterations
    fps = 1.0 / avg_time
    
    return {
        "avg_time_ms": avg_time * 1000,
        "fps": fps,
        "total_time": elapsed_time,
        "iterations": num_iterations
    }


def test_inference_speed_comparison():
    """Compare inference speed across different architectures."""
    print("\n=== Test 1: Inference Speed Comparison ===")
    
    environments = [
        ("Pendulum-v1", 3, 1),
        ("HalfCheetah-v5", 17, 6),
    ]
    
    results = {}
    
    for env_name, obs_dim, action_dim in environments:
        print(f"\n  Environment: {env_name} (obs={obs_dim}, action={action_dim})")
        
        # Create models
        config = DHCSSMConfig(hidden_dim=64)
        dhc_policy = RLPolicyAdapter(obs_dim, action_dim, config)
        mlp_policy = MLPPolicy(obs_dim, action_dim, hidden_dim=128)
        cnn_policy = CNNPolicy(obs_dim, action_dim, hidden_dim=64)
        
        # Count parameters
        dhc_params = count_parameters(dhc_policy)
        mlp_params = count_parameters(mlp_policy)
        cnn_params = count_parameters(cnn_policy)
        
        print(f"  Parameters: DHC={dhc_params:,}, MLP={mlp_params:,}, CNN={cnn_params:,}")
        
        # Create input
        obs_tensor = torch.randn(1, obs_dim)
        
        # Measure speed
        dhc_speed = measure_inference_speed(dhc_policy, obs_tensor)
        mlp_speed = measure_inference_speed(mlp_policy, obs_tensor)
        cnn_speed = measure_inference_speed(cnn_policy, obs_tensor)
        
        print(f"  DHC-SSM: {dhc_speed['avg_time_ms']:.3f}ms ({dhc_speed['fps']:.1f} FPS)")
        print(f"  MLP:     {mlp_speed['avg_time_ms']:.3f}ms ({mlp_speed['fps']:.1f} FPS)")
        print(f"  CNN:     {cnn_speed['avg_time_ms']:.3f}ms ({cnn_speed['fps']:.1f} FPS)")
        
        # Calculate speedup
        dhc_vs_mlp = mlp_speed['avg_time_ms'] / dhc_speed['avg_time_ms']
        dhc_vs_cnn = cnn_speed['avg_time_ms'] / dhc_speed['avg_time_ms']
        
        print(f"  Speedup: DHC vs MLP = {dhc_vs_mlp:.2f}x, DHC vs CNN = {dhc_vs_cnn:.2f}x")
        
        results[env_name] = {
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "dhc_ssm": {
                "params": dhc_params,
                "time_ms": dhc_speed['avg_time_ms'],
                "fps": dhc_speed['fps']
            },
            "mlp": {
                "params": mlp_params,
                "time_ms": mlp_speed['avg_time_ms'],
                "fps": mlp_speed['fps']
            },
            "cnn": {
                "params": cnn_params,
                "time_ms": cnn_speed['avg_time_ms'],
                "fps": cnn_speed['fps']
            },
            "speedup_vs_mlp": dhc_vs_mlp,
            "speedup_vs_cnn": dhc_vs_cnn
        }
    
    return True, results


def test_parameter_efficiency():
    """Compare parameter efficiency (performance per parameter)."""
    print("\n=== Test 2: Parameter Efficiency ===")
    
    obs_dim, action_dim = 17, 6  # HalfCheetah
    
    # Create models with similar parameter counts
    configs = [
        ("DHC-SSM (hidden=32)", lambda: RLPolicyAdapter(obs_dim, action_dim, 
                                                        DHCSSMConfig(hidden_dim=32))),
        ("DHC-SSM (hidden=64)", lambda: RLPolicyAdapter(obs_dim, action_dim, 
                                                        DHCSSMConfig(hidden_dim=64))),
        ("MLP (hidden=64)", lambda: MLPPolicy(obs_dim, action_dim, hidden_dim=64)),
        ("MLP (hidden=128)", lambda: MLPPolicy(obs_dim, action_dim, hidden_dim=128)),
        ("MLP (hidden=256)", lambda: MLPPolicy(obs_dim, action_dim, hidden_dim=256)),
    ]
    
    results = []
    
    for name, model_fn in configs:
        model = model_fn()
        params = count_parameters(model)
        
        obs_tensor = torch.randn(1, obs_dim)
        speed = measure_inference_speed(model, obs_tensor, num_iterations=500)
        
        print(f"  {name:25s}: {params:7,} params, {speed['avg_time_ms']:6.3f}ms, {speed['fps']:7.1f} FPS")
        
        results.append({
            "name": name,
            "params": params,
            "time_ms": speed['avg_time_ms'],
            "fps": speed['fps']
        })
    
    return True, results


def test_batch_processing():
    """Compare batch processing efficiency."""
    print("\n=== Test 3: Batch Processing Efficiency ===")
    
    obs_dim, action_dim = 17, 6
    batch_sizes = [1, 4, 16, 32, 64]
    
    config = DHCSSMConfig(hidden_dim=64)
    dhc_policy = RLPolicyAdapter(obs_dim, action_dim, config)
    mlp_policy = MLPPolicy(obs_dim, action_dim, hidden_dim=128)
    
    results = {}
    
    for batch_size in batch_sizes:
        obs_tensor = torch.randn(batch_size, obs_dim)
        
        dhc_speed = measure_inference_speed(dhc_policy, obs_tensor, num_iterations=200)
        mlp_speed = measure_inference_speed(mlp_policy, obs_tensor, num_iterations=200)
        
        dhc_throughput = batch_size * dhc_speed['fps']
        mlp_throughput = batch_size * mlp_speed['fps']
        
        print(f"  Batch={batch_size:3d}: DHC={dhc_throughput:8.1f} samples/s, "
              f"MLP={mlp_throughput:8.1f} samples/s")
        
        results[f"batch_{batch_size}"] = {
            "dhc_throughput": dhc_throughput,
            "mlp_throughput": mlp_throughput,
            "dhc_time_ms": dhc_speed['avg_time_ms'],
            "mlp_time_ms": mlp_speed['avg_time_ms']
        }
    
    return True, results


def test_memory_usage():
    """Compare memory usage during forward pass."""
    print("\n=== Test 4: Memory Usage Comparison ===")
    
    obs_dim, action_dim = 17, 6
    batch_size = 32
    
    config = DHCSSMConfig(hidden_dim=64)
    dhc_policy = RLPolicyAdapter(obs_dim, action_dim, config)
    mlp_policy = MLPPolicy(obs_dim, action_dim, hidden_dim=128)
    
    obs_tensor = torch.randn(batch_size, obs_dim)
    
    # Measure model size
    dhc_params = count_parameters(dhc_policy)
    mlp_params = count_parameters(mlp_policy)
    
    dhc_size_mb = dhc_params * 4 / (1024 ** 2)  # 4 bytes per float32
    mlp_size_mb = mlp_params * 4 / (1024 ** 2)
    
    print(f"  DHC-SSM: {dhc_params:,} params ({dhc_size_mb:.2f} MB)")
    print(f"  MLP:     {mlp_params:,} params ({mlp_size_mb:.2f} MB)")
    
    results = {
        "dhc_params": dhc_params,
        "mlp_params": mlp_params,
        "dhc_size_mb": dhc_size_mb,
        "mlp_size_mb": mlp_size_mb
    }
    
    return True, results


def run_all_baseline_tests():
    """Run all baseline comparison tests."""
    print("=" * 70)
    print("DHC-SSM v3.1 - BASELINE COMPARISON TESTS")
    print("=" * 70)
    
    all_results = {}
    
    tests = [
        ("Inference Speed Comparison", test_inference_speed_comparison),
        ("Parameter Efficiency", test_parameter_efficiency),
        ("Batch Processing Efficiency", test_batch_processing),
        ("Memory Usage Comparison", test_memory_usage),
    ]
    
    passed = 0
    
    for test_name, test_func in tests:
        try:
            success, result = test_func()
            all_results[test_name] = {"passed": success, "data": result}
            if success:
                passed += 1
        except Exception as e:
            print(f"\n=== {test_name} ===\n✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            all_results[test_name] = {"passed": False, "error": str(e)}
    
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON RESULTS")
    print("=" * 70)
    
    for test_name, result in all_results.items():
        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{len(tests)} tests passed")
    
    # Save results
    results_file = "tests/baseline_comparison_results.json"
    try:
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to: {results_file}")
    except Exception as e:
        print(f"\nWarning: Could not save results to JSON: {e}")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = run_all_baseline_tests()
    exit(0 if success else 1)

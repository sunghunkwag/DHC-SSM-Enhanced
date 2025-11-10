"""
MuJoCo Benchmark Tests for DHC-SSM Architecture.

This module tests the DHC-SSM model on reinforcement learning control tasks
using MuJoCo physics simulation environments.
"""

import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import time
import sys
import os
import json
from typing import Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dhc_ssm.core.model import DHCSSMModel, DHCSSMConfig
from dhc_ssm.adapters.rl_policy import RLPolicyAdapter as ImprovedRLPolicyAdapter


class RLPolicyAdapter(nn.Module):
    """Adapter to use DHC-SSM as a policy network for RL tasks."""
    
    def __init__(self, observation_dim: int, action_dim: int, config: DHCSSMConfig):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        
        # Reshape observation to image-like format for DHC-SSM
        # We'll create a square-ish representation
        self.obs_reshape_dim = int(np.ceil(np.sqrt(observation_dim)))
        self.pad_size = self.obs_reshape_dim ** 2 - observation_dim
        
        # Update config for RL task
        config.input_channels = 1
        config.output_dim = action_dim
        
        self.dhc_model = DHCSSMModel(config)
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass converting observations to actions.
        
        Args:
            obs: Observation tensor of shape (batch_size, observation_dim)
            
        Returns:
            Action tensor of shape (batch_size, action_dim)
        """
        batch_size = obs.shape[0]
        
        # Pad and reshape observation
        if self.pad_size > 0:
            obs_padded = torch.cat([obs, torch.zeros(batch_size, self.pad_size, device=obs.device)], dim=1)
        else:
            obs_padded = obs
            
        # Reshape to image format: (batch, 1, H, W)
        obs_reshaped = obs_padded.view(batch_size, 1, self.obs_reshape_dim, self.obs_reshape_dim)
        
        # Forward through DHC-SSM
        actions = self.dhc_model(obs_reshaped)
        
        return torch.tanh(actions)  # Bound actions to [-1, 1]


def test_mujoco_environment_compatibility():
    """Test compatibility with various MuJoCo environments."""
    print("\n=== Test 1: MuJoCo Environment Compatibility ===")
    
    environments = [
        "Pendulum-v1",
        "HalfCheetah-v5",
        "Hopper-v5",
        "Walker2d-v5",
    ]
    
    results = {}
    
    for env_name in environments:
        try:
            env = gym.make(env_name)
            obs_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            
            # Create policy
            config = DHCSSMConfig()
            policy = RLPolicyAdapter(obs_dim, action_dim, config)
            
            # Test forward pass
            obs, _ = env.reset()
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            with torch.no_grad():
                action = policy(obs_tensor)
            
            assert action.shape == (1, action_dim), f"Action shape mismatch for {env_name}"
            
            results[env_name] = "✓ PASS"
            print(f"  {env_name}: ✓ Compatible (obs_dim={obs_dim}, action_dim={action_dim})")
            
            env.close()
            
        except Exception as e:
            results[env_name] = f"✗ FAIL: {str(e)}"
            print(f"  {env_name}: ✗ Failed - {e}")
    
    success = all("✓" in v for v in results.values())
    return success, results


def test_policy_rollout():
    """Test policy rollout in MuJoCo environment."""
    print("\n=== Test 2: Policy Rollout ===")
    
    env_name = "Pendulum-v1"
    env = gym.make(env_name)
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    try:
        # Create policy
        config = DHCSSMConfig()
        policy = RLPolicyAdapter(obs_dim, action_dim, config)
        policy.eval()
        
        # Run episode
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 200
        
        with torch.no_grad():
            for _ in range(max_steps):
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action = policy(obs_tensor).squeeze(0).numpy()
                
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
                
                if terminated or truncated:
                    break
        
        env.close()
        
        print(f"✓ Rollout successful")
        print(f"  Environment: {env_name}")
        print(f"  Steps: {steps}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Average Reward: {total_reward/steps:.2f}")
        
        return True, {"env": env_name, "steps": steps, "total_reward": float(total_reward)}
        
    except Exception as e:
        print(f"✗ Rollout failed: {e}")
        env.close()
        return False, {}


def test_training_on_mujoco():
    """Test if the model can learn from MuJoCo environment data."""
    print("\n=== Test 3: Learning from MuJoCo Data ===")
    
    env_name = "Pendulum-v1"
    env = gym.make(env_name)
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    try:
        # Create policy
        config = DHCSSMConfig()
        policy = RLPolicyAdapter(obs_dim, action_dim, config)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        
        # Collect some experience
        experiences = []
        obs, _ = env.reset()
        
        for _ in range(100):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            experiences.append((obs, action, reward, next_obs))
            obs = next_obs
            
            if terminated or truncated:
                obs, _ = env.reset()
        
        # Train on collected data
        policy.train()
        initial_losses = []
        final_losses = []
        
        # Initial loss
        for obs, action, _, _ in experiences[:20]:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action_tensor = torch.FloatTensor(action).unsqueeze(0)
            
            pred_action = policy(obs_tensor)
            loss = nn.functional.mse_loss(pred_action, action_tensor)
            initial_losses.append(loss.item())
        
        # Training
        for epoch in range(10):
            for obs, action, _, _ in experiences:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action_tensor = torch.FloatTensor(action).unsqueeze(0)
                
                optimizer.zero_grad()
                pred_action = policy(obs_tensor)
                loss = nn.functional.mse_loss(pred_action, action_tensor)
                loss.backward()
                optimizer.step()
        
        # Final loss
        policy.eval()
        with torch.no_grad():
            for obs, action, _, _ in experiences[:20]:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action_tensor = torch.FloatTensor(action).unsqueeze(0)
                
                pred_action = policy(obs_tensor)
                loss = nn.functional.mse_loss(pred_action, action_tensor)
                final_losses.append(loss.item())
        
        initial_avg = np.mean(initial_losses)
        final_avg = np.mean(final_losses)
        improvement = initial_avg - final_avg
        
        env.close()
        
        print(f"✓ Learning test completed")
        print(f"  Initial Loss: {initial_avg:.4f}")
        print(f"  Final Loss: {final_avg:.4f}")
        print(f"  Improvement: {improvement:.4f}")
        
        return improvement > 0, {
            "initial_loss": float(initial_avg),
            "final_loss": float(final_avg),
            "improvement": float(improvement)
        }
        
    except Exception as e:
        print(f"✗ Learning test failed: {e}")
        env.close()
        return False, {}


def test_performance_metrics():
    """Test performance metrics on MuJoCo tasks."""
    print("\n=== Test 4: Performance Metrics ===")
    
    env_name = "Pendulum-v1"
    env = gym.make(env_name)
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    try:
        # Create policy
        config = DHCSSMConfig()
        policy = RLPolicyAdapter(obs_dim, action_dim, config)
        policy.eval()
        
        # Measure inference time
        obs, _ = env.reset()
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = policy(obs_tensor)
        
        # Benchmark
        num_inferences = 1000
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_inferences):
                _ = policy(obs_tensor)
        
        elapsed_time = time.time() - start_time
        avg_inference_time = elapsed_time / num_inferences
        fps = 1.0 / avg_inference_time
        
        # Count parameters
        num_params = sum(p.numel() for p in policy.parameters())
        
        env.close()
        
        print(f"✓ Performance metrics collected")
        print(f"  Average Inference Time: {avg_inference_time*1000:.3f} ms")
        print(f"  Throughput: {fps:.1f} FPS")
        print(f"  Total Parameters: {num_params:,}")
        
        return True, {
            "inference_time_ms": float(avg_inference_time * 1000),
            "fps": float(fps),
            "num_parameters": int(num_params)
        }
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        env.close()
        return False, {}


def test_multi_environment_benchmark():
    """Benchmark across multiple MuJoCo environments."""
    print("\n=== Test 5: Multi-Environment Benchmark ===")
    
    environments = ["Pendulum-v1", "HalfCheetah-v5", "Hopper-v5"]
    results = {}
    
    for env_name in environments:
        try:
            env = gym.make(env_name)
            obs_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            
            # Create policy
            config = DHCSSMConfig()
            policy = RLPolicyAdapter(obs_dim, action_dim, config)
            policy.eval()
            
            # Run episode
            obs, _ = env.reset()
            total_reward = 0
            steps = 0
            max_steps = 200
            
            with torch.no_grad():
                for _ in range(max_steps):
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    action = policy(obs_tensor).squeeze(0).numpy()
                    
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total_reward += reward
                    steps += 1
                    
                    if terminated or truncated:
                        break
            
            results[env_name] = {
                "steps": steps,
                "total_reward": float(total_reward),
                "avg_reward": float(total_reward / steps)
            }
            
            print(f"  {env_name}: {steps} steps, reward={total_reward:.2f}")
            
            env.close()
            
        except Exception as e:
            print(f"  {env_name}: ✗ Failed - {e}")
            results[env_name] = {"error": str(e)}
    
    success = all("error" not in v for v in results.values())
    print(f"✓ Multi-environment benchmark completed")
    
    return success, results


def run_all_mujoco_tests():
    """Run all MuJoCo benchmark tests."""
    print("=" * 70)
    print("DHC-SSM v3.1 - MUJOCO BENCHMARK")
    print("=" * 70)
    
    all_results = {}
    
    tests = [
        ("Environment Compatibility", test_mujoco_environment_compatibility),
        ("Policy Rollout", test_policy_rollout),
        ("Learning from MuJoCo Data", test_training_on_mujoco),
        ("Performance Metrics", test_performance_metrics),
        ("Multi-Environment Benchmark", test_multi_environment_benchmark),
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
            all_results[test_name] = {"passed": False, "error": str(e)}
    
    print("\n" + "=" * 70)
    print("MUJOCO BENCHMARK RESULTS")
    print("=" * 70)
    
    for test_name, result in all_results.items():
        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{len(tests)} tests passed")
    
    # Save results
    results_file = "tests/mujoco_benchmark_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    if passed == len(tests):
        print("\n🎉 ALL MUJOCO TESTS PASSED - DHC-SSM is RL-ready!")
        return True
    else:
        print(f"\n⚠️  {len(tests) - passed} tests failed - needs attention")
        return False


if __name__ == "__main__":
    success = run_all_mujoco_tests()
    exit(0 if success else 1)

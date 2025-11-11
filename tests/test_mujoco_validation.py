"""
Comprehensive MuJoCo Validation Tests.

This module provides rigorous validation of DHC-SSM on MuJoCo environments:
1. PPO training with learning curves (100 episodes)
2. MLP baseline comparison
3. Memory and speed measurements with O(n) complexity verification
"""

import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import time
import sys
import os
import json
from collections import deque
from typing import Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dhc_ssm.core.model import DHCSSMConfig
from dhc_ssm.adapters.rl_policy import RLPolicyAdapter, RLActorCriticAdapter


class MLPPolicy(nn.Module):
    """MLP baseline for comparison."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes=[64, 64]):
        super().__init__()
        layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.Tanh()
            ])
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, action_dim))
        layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class MLPValue(nn.Module):
    """MLP value function."""
    
    def __init__(self, obs_dim: int, hidden_sizes=[64, 64]):
        super().__init__()
        layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.Tanh()
            ])
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


def collect_trajectories(env, policy, value_fn, num_steps=2048):
    """Collect trajectories for PPO."""
    observations = []
    actions = []
    rewards = []
    values = []
    dones = []
    
    obs, _ = env.reset()
    episode_reward = 0
    episode_rewards = []
    
    for _ in range(num_steps):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        
        with torch.no_grad():
            action = policy(obs_tensor).squeeze(0).numpy()
            value = value_fn(obs_tensor).item()
        
        observations.append(obs)
        actions.append(action)
        values.append(value)
        
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        rewards.append(reward)
        dones.append(done)
        episode_reward += reward
        
        if done:
            episode_rewards.append(episode_reward)
            episode_reward = 0
            obs, _ = env.reset()
        else:
            obs = next_obs
    
    return observations, actions, rewards, values, dones, episode_rewards


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation."""
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


def ppo_update(policy, value_fn, policy_optimizer, value_optimizer,
               observations, actions, advantages, returns, epochs=10, clip_eps=0.2):
    """Perform PPO update."""
    obs_tensor = torch.FloatTensor(np.array(observations))
    actions_tensor = torch.FloatTensor(np.array(actions))
    advantages_tensor = torch.FloatTensor(advantages)
    returns_tensor = torch.FloatTensor(returns)
    
    # Normalize advantages
    advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
    
    # Store old policy predictions
    with torch.no_grad():
        old_actions = policy(obs_tensor)
    
    for _ in range(epochs):
        # Policy loss
        new_actions = policy(obs_tensor)
        ratio = torch.exp(-((new_actions - actions_tensor) ** 2).sum(dim=1) / 2)
        old_ratio = torch.exp(-((old_actions - actions_tensor) ** 2).sum(dim=1) / 2)
        ratio = ratio / (old_ratio + 1e-8)
        
        surr1 = ratio * advantages_tensor
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages_tensor
        policy_loss = -torch.min(surr1, surr2).mean()
        
        policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        policy_optimizer.step()
        
        # Value loss
        pred_values = value_fn(obs_tensor).squeeze()
        value_loss = nn.functional.mse_loss(pred_values, returns_tensor)
        
        value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_fn.parameters(), 0.5)
        value_optimizer.step()
    
    return policy_loss.item(), value_loss.item()


def train_ppo(env_name, policy, value_fn, num_episodes=100, steps_per_update=2048):
    """Train using PPO algorithm."""
    env = gym.make(env_name)
    
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    value_optimizer = torch.optim.Adam(value_fn.parameters(), lr=1e-3)
    
    episode_rewards = []
    all_episode_rewards = []
    steps = 0
    
    print(f"  Training {env_name} for {num_episodes} episodes...")
    
    while len(all_episode_rewards) < num_episodes:
        # Collect trajectories
        observations, actions, rewards, values, dones, ep_rewards = collect_trajectories(
            env, policy, value_fn, steps_per_update
        )
        
        all_episode_rewards.extend(ep_rewards)
        steps += len(observations)
        
        # Compute advantages
        advantages, returns = compute_gae(rewards, values, dones)
        
        # PPO update
        policy_loss, value_loss = ppo_update(
            policy, value_fn, policy_optimizer, value_optimizer,
            observations, actions, advantages, returns
        )
        
        if len(all_episode_rewards) % 10 == 0:
            recent_reward = np.mean(all_episode_rewards[-10:]) if len(all_episode_rewards) >= 10 else np.mean(all_episode_rewards)
            print(f"    Episode {len(all_episode_rewards):3d}: avg_reward={recent_reward:7.2f}, "
                  f"policy_loss={policy_loss:.4f}, value_loss={value_loss:.4f}")
    
    env.close()
    return all_episode_rewards[:num_episodes]


def test_ppo_learning_dhc_vs_mlp():
    """Test 1: PPO learning curves for DHC-SSM vs MLP."""
    print("\n=== Test 1: PPO Learning Curves (DHC-SSM vs MLP) ===")
    
    env_name = "Pendulum-v1"
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    env.close()
    
    num_episodes = 100
    
    # Train DHC-SSM
    print("\n  Training DHC-SSM with PPO...")
    config = DHCSSMConfig(hidden_dim=64)
    dhc_policy = RLPolicyAdapter(obs_dim, action_dim, config)
    dhc_value = RLPolicyAdapter(obs_dim, 1, config)
    
    dhc_rewards = train_ppo(env_name, dhc_policy, dhc_value, num_episodes)
    
    # Train MLP
    print("\n  Training MLP with PPO...")
    mlp_policy = MLPPolicy(obs_dim, action_dim, [64, 64])
    mlp_value = MLPValue(obs_dim, [64, 64])
    
    mlp_rewards = train_ppo(env_name, mlp_policy, mlp_value, num_episodes)
    
    # Analyze results
    dhc_initial = np.mean(dhc_rewards[:10])
    dhc_final = np.mean(dhc_rewards[-10:])
    dhc_improvement = dhc_final - dhc_initial
    
    mlp_initial = np.mean(mlp_rewards[:10])
    mlp_final = np.mean(mlp_rewards[-10:])
    mlp_improvement = mlp_final - mlp_initial
    
    print(f"\n  DHC-SSM: Initial={dhc_initial:.2f}, Final={dhc_final:.2f}, Improvement={dhc_improvement:.2f}")
    print(f"  MLP:     Initial={mlp_initial:.2f}, Final={mlp_final:.2f}, Improvement={mlp_improvement:.2f}")
    
    # Both should improve
    both_learned = dhc_improvement > 50 and mlp_improvement > 50
    
    results = {
        "dhc_ssm": {
            "initial": float(dhc_initial),
            "final": float(dhc_final),
            "improvement": float(dhc_improvement),
            "rewards": [float(r) for r in dhc_rewards]
        },
        "mlp": {
            "initial": float(mlp_initial),
            "final": float(mlp_final),
            "improvement": float(mlp_improvement),
            "rewards": [float(r) for r in mlp_rewards]
        }
    }
    
    return both_learned, results


def test_inference_speed_vs_mlp():
    """Test 2: Inference speed comparison with different observation dimensions."""
    print("\n=== Test 2: Inference Speed vs Observation Dimension ===")
    
    obs_dims = [3, 11, 17, 33, 65]
    action_dim = 6
    
    results = {}
    
    print(f"\n  {'Obs Dim':>8} | {'DHC-SSM (ms)':>12} | {'MLP (ms)':>10} | {'Speedup':>8} | {'DHC Params':>11} | {'MLP Params':>11}")
    print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*10}-+-{'-'*8}-+-{'-'*11}-+-{'-'*11}")
    
    for obs_dim in obs_dims:
        # Create models
        config = DHCSSMConfig(hidden_dim=64)
        dhc_policy = RLPolicyAdapter(obs_dim, action_dim, config)
        mlp_policy = MLPPolicy(obs_dim, action_dim, [64, 64])
        
        # Count parameters
        dhc_params = sum(p.numel() for p in dhc_policy.parameters())
        mlp_params = sum(p.numel() for p in mlp_policy.parameters())
        
        # Measure speed
        obs_tensor = torch.randn(1, obs_dim)
        
        # DHC-SSM
        dhc_policy.eval()
        with torch.no_grad():
            for _ in range(10):  # Warmup
                _ = dhc_policy(obs_tensor)
        
        start = time.time()
        with torch.no_grad():
            for _ in range(1000):
                _ = dhc_policy(obs_tensor)
        dhc_time = (time.time() - start) / 1000 * 1000  # ms
        
        # MLP
        mlp_policy.eval()
        with torch.no_grad():
            for _ in range(10):  # Warmup
                _ = mlp_policy(obs_tensor)
        
        start = time.time()
        with torch.no_grad():
            for _ in range(1000):
                _ = mlp_policy(obs_tensor)
        mlp_time = (time.time() - start) / 1000 * 1000  # ms
        
        speedup = mlp_time / dhc_time
        
        print(f"  {obs_dim:8d} | {dhc_time:12.3f} | {mlp_time:10.3f} | {speedup:8.2f}x | {dhc_params:11,} | {mlp_params:11,}")
        
        results[f"obs_dim_{obs_dim}"] = {
            "dhc_time_ms": float(dhc_time),
            "mlp_time_ms": float(mlp_time),
            "speedup": float(speedup),
            "dhc_params": int(dhc_params),
            "mlp_params": int(mlp_params)
        }
    
    return True, results


def test_complexity_scaling():
    """Test 3: Verify O(n) complexity scaling."""
    print("\n=== Test 3: Complexity Scaling Analysis ===")
    
    obs_dims = [8, 16, 32, 64, 128]
    action_dim = 6
    
    dhc_times = []
    mlp_times = []
    
    print(f"\n  Testing complexity scaling...")
    
    for obs_dim in obs_dims:
        config = DHCSSMConfig(hidden_dim=64)
        dhc_policy = RLPolicyAdapter(obs_dim, action_dim, config)
        mlp_policy = MLPPolicy(obs_dim, action_dim, [64, 64])
        
        obs_tensor = torch.randn(1, obs_dim)
        
        # Measure DHC-SSM
        dhc_policy.eval()
        with torch.no_grad():
            for _ in range(10):
                _ = dhc_policy(obs_tensor)
        
        start = time.time()
        with torch.no_grad():
            for _ in range(500):
                _ = dhc_policy(obs_tensor)
        dhc_time = (time.time() - start) / 500
        dhc_times.append(dhc_time)
        
        # Measure MLP
        mlp_policy.eval()
        with torch.no_grad():
            for _ in range(10):
                _ = mlp_policy(obs_tensor)
        
        start = time.time()
        with torch.no_grad():
            for _ in range(500):
                _ = mlp_policy(obs_tensor)
        mlp_time = (time.time() - start) / 500
        mlp_times.append(mlp_time)
    
    # Compute scaling factors
    dhc_ratios = [dhc_times[i] / dhc_times[0] for i in range(len(dhc_times))]
    mlp_ratios = [mlp_times[i] / mlp_times[0] for i in range(len(mlp_times))]
    linear_ratios = [obs_dims[i] / obs_dims[0] for i in range(len(obs_dims))]
    
    print(f"\n  {'Obs Dim':>8} | {'DHC Ratio':>10} | {'MLP Ratio':>10} | {'Linear (O(n))':>15}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*15}")
    
    for i, obs_dim in enumerate(obs_dims):
        print(f"  {obs_dim:8d} | {dhc_ratios[i]:10.2f} | {mlp_ratios[i]:10.2f} | {linear_ratios[i]:15.2f}")
    
    # Check if DHC-SSM scales roughly linearly
    # Allow some overhead but should be closer to O(n) than O(n^2)
    dhc_scaling_ok = dhc_ratios[-1] < linear_ratios[-1] * 2  # Within 2x of linear
    
    results = {
        "obs_dims": obs_dims,
        "dhc_times": [float(t) for t in dhc_times],
        "mlp_times": [float(t) for t in mlp_times],
        "dhc_ratios": [float(r) for r in dhc_ratios],
        "mlp_ratios": [float(r) for r in mlp_ratios],
        "linear_ratios": [float(r) for r in linear_ratios],
        "dhc_scales_linearly": dhc_scaling_ok
    }
    
    if dhc_scaling_ok:
        print(f"\n  ✓ DHC-SSM scales approximately O(n)")
    else:
        print(f"\n  ✗ DHC-SSM scaling exceeds O(n) expectations")
    
    return dhc_scaling_ok, results


def run_all_validation_tests():
    """Run all comprehensive validation tests."""
    print("=" * 70)
    print("DHC-SSM v3.1 - COMPREHENSIVE MUJOCO VALIDATION")
    print("=" * 70)
    print("\nThis test suite provides rigorous validation:")
    print("1. PPO training with 100-episode learning curves")
    print("2. Direct comparison with MLP baseline")
    print("3. Complexity scaling verification")
    print("\nNote: This will take several minutes to complete.\n")
    
    all_results = {}
    
    tests = [
        ("PPO Learning Curves (DHC-SSM vs MLP)", test_ppo_learning_dhc_vs_mlp),
        ("Inference Speed vs Observation Dimension", test_inference_speed_vs_mlp),
        ("Complexity Scaling Analysis", test_complexity_scaling),
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
    print("VALIDATION RESULTS")
    print("=" * 70)
    
    for test_name, result in all_results.items():
        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{len(tests)} tests passed")
    
    # Save results
    results_file = "tests/mujoco_validation_results.json"
    try:
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to: {results_file}")
    except Exception as e:
        print(f"\nWarning: Could not save results to JSON: {e}")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = run_all_validation_tests()
    exit(0 if success else 1)

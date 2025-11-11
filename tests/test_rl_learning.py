"""
RL Learning Validation Test.

This module tests whether DHC-SSM can actually learn from RL experience,
not just perform forward passes. Uses a simple policy gradient approach.
"""

import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import sys
import os
import json
from typing import List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dhc_ssm.core.model import DHCSSMConfig
from dhc_ssm.adapters.rl_policy import RLPolicyAdapter


def collect_episode(env, policy, max_steps=200):
    """Collect a single episode of experience."""
    observations = []
    actions = []
    rewards = []
    
    obs, _ = env.reset()
    
    for _ in range(max_steps):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        
        with torch.no_grad():
            action = policy(obs_tensor).squeeze(0).numpy()
        
        observations.append(obs)
        actions.append(action)
        
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        
        if terminated or truncated:
            break
    
    return observations, actions, rewards


def compute_returns(rewards, gamma=0.99):
    """Compute discounted returns."""
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def simple_policy_gradient_step(policy, optimizer, observations, actions, returns):
    """Perform a simple policy gradient update."""
    # Convert to tensors
    obs_tensor = torch.FloatTensor(np.array(observations))
    actions_tensor = torch.FloatTensor(np.array(actions))
    returns_tensor = torch.FloatTensor(returns)
    
    # Normalize returns
    returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
    
    # Forward pass
    pred_actions = policy(obs_tensor)
    
    # Policy gradient loss (simplified - using MSE as proxy)
    loss = nn.functional.mse_loss(pred_actions, actions_tensor, reduction='none')
    loss = (loss.mean(dim=1) * returns_tensor).mean()
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
    optimizer.step()
    
    return loss.item()


def test_simple_learning():
    """Test if DHC-SSM can learn to improve on Pendulum."""
    print("\n=== Test 1: Simple Policy Gradient Learning ===")
    
    env = gym.make('Pendulum-v1')
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create policy
    config = DHCSSMConfig(hidden_dim=64)
    policy = RLPolicyAdapter(obs_dim, action_dim, config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    
    # Training
    num_episodes = 50
    episode_rewards = []
    
    print(f"  Training for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        # Collect episode
        observations, actions, rewards = collect_episode(env, policy)
        episode_reward = sum(rewards)
        episode_rewards.append(episode_reward)
        
        # Compute returns
        returns = compute_returns(rewards)
        
        # Update policy
        loss = simple_policy_gradient_step(policy, optimizer, observations, actions, returns)
        
        if (episode + 1) % 10 == 0:
            recent_avg = np.mean(episode_rewards[-10:])
            print(f"  Episode {episode+1:3d}: reward={episode_reward:7.2f}, "
                  f"avg_10={recent_avg:7.2f}, loss={loss:.4f}")
    
    env.close()
    
    # Analyze learning
    initial_avg = np.mean(episode_rewards[:10])
    final_avg = np.mean(episode_rewards[-10:])
    improvement = final_avg - initial_avg
    
    print(f"\n  Initial performance (first 10 episodes): {initial_avg:.2f}")
    print(f"  Final performance (last 10 episodes):    {final_avg:.2f}")
    print(f"  Improvement: {improvement:.2f}")
    
    # Learning is successful if improvement is positive
    learned = improvement > 0
    
    if learned:
        print(f"  ✓ Learning verified: Policy improved over training")
    else:
        print(f"  ✗ No learning detected: Policy did not improve")
    
    results = {
        "initial_avg": float(initial_avg),
        "final_avg": float(final_avg),
        "improvement": float(improvement),
        "learned": learned,
        "episode_rewards": [float(r) for r in episode_rewards]
    }
    
    return learned, results


def test_learning_vs_random():
    """Compare learned policy vs random policy."""
    print("\n=== Test 2: Learned vs Random Policy ===")
    
    env = gym.make('Pendulum-v1')
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Train a policy
    config = DHCSSMConfig(hidden_dim=64)
    trained_policy = RLPolicyAdapter(obs_dim, action_dim, config)
    optimizer = torch.optim.Adam(trained_policy.parameters(), lr=3e-4)
    
    print("  Training policy...")
    for episode in range(30):
        observations, actions, rewards = collect_episode(env, trained_policy)
        returns = compute_returns(rewards)
        simple_policy_gradient_step(trained_policy, optimizer, observations, actions, returns)
    
    # Evaluate trained policy
    print("  Evaluating trained policy...")
    trained_rewards = []
    for _ in range(10):
        _, _, rewards = collect_episode(env, trained_policy)
        trained_rewards.append(sum(rewards))
    trained_avg = np.mean(trained_rewards)
    
    # Evaluate random policy
    print("  Evaluating random policy...")
    random_policy = RLPolicyAdapter(obs_dim, action_dim, config)  # Untrained
    random_rewards = []
    for _ in range(10):
        _, _, rewards = collect_episode(env, random_policy)
        random_rewards.append(sum(rewards))
    random_avg = np.mean(random_rewards)
    
    env.close()
    
    print(f"\n  Trained policy average: {trained_avg:.2f}")
    print(f"  Random policy average:  {random_avg:.2f}")
    print(f"  Difference: {trained_avg - random_avg:.2f}")
    
    better_than_random = trained_avg > random_avg
    
    if better_than_random:
        print(f"  ✓ Trained policy is better than random")
    else:
        print(f"  ✗ Trained policy is not better than random")
    
    results = {
        "trained_avg": float(trained_avg),
        "random_avg": float(random_avg),
        "difference": float(trained_avg - random_avg),
        "better_than_random": better_than_random
    }
    
    return better_than_random, results


def test_learning_stability():
    """Test if learning is stable across multiple runs."""
    print("\n=== Test 3: Learning Stability ===")
    
    env = gym.make('Pendulum-v1')
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    num_runs = 3
    all_improvements = []
    
    for run in range(num_runs):
        print(f"\n  Run {run+1}/{num_runs}")
        
        # Create fresh policy
        config = DHCSSMConfig(hidden_dim=64)
        policy = RLPolicyAdapter(obs_dim, action_dim, config)
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        
        # Train
        episode_rewards = []
        for episode in range(30):
            observations, actions, rewards = collect_episode(env, policy)
            episode_rewards.append(sum(rewards))
            returns = compute_returns(rewards)
            simple_policy_gradient_step(policy, optimizer, observations, actions, returns)
        
        initial_avg = np.mean(episode_rewards[:5])
        final_avg = np.mean(episode_rewards[-5:])
        improvement = final_avg - initial_avg
        all_improvements.append(improvement)
        
        print(f"    Initial: {initial_avg:.2f}, Final: {final_avg:.2f}, "
              f"Improvement: {improvement:.2f}")
    
    env.close()
    
    avg_improvement = np.mean(all_improvements)
    std_improvement = np.std(all_improvements)
    consistent = all(imp > 0 for imp in all_improvements)
    
    print(f"\n  Average improvement: {avg_improvement:.2f} ± {std_improvement:.2f}")
    print(f"  All runs improved: {consistent}")
    
    if consistent:
        print(f"  ✓ Learning is consistent across runs")
    else:
        print(f"  ✗ Learning is inconsistent")
    
    results = {
        "improvements": [float(imp) for imp in all_improvements],
        "avg_improvement": float(avg_improvement),
        "std_improvement": float(std_improvement),
        "consistent": consistent
    }
    
    return consistent, results


def run_all_learning_tests():
    """Run all RL learning tests."""
    print("=" * 70)
    print("DHC-SSM v3.1 - RL LEARNING VALIDATION TESTS")
    print("=" * 70)
    print("\nNote: These tests use simple policy gradient (not PPO/SAC)")
    print("to verify basic learning capability.\n")
    
    all_results = {}
    
    tests = [
        ("Simple Policy Gradient Learning", test_simple_learning),
        ("Learned vs Random Policy", test_learning_vs_random),
        ("Learning Stability", test_learning_stability),
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
    print("RL LEARNING VALIDATION RESULTS")
    print("=" * 70)
    
    for test_name, result in all_results.items():
        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{len(tests)} tests passed")
    
    # Save results
    results_file = "tests/rl_learning_results.json"
    try:
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to: {results_file}")
    except Exception as e:
        print(f"\nWarning: Could not save results to JSON: {e}")
    
    print("\n" + "=" * 70)
    print("IMPORTANT NOTES")
    print("=" * 70)
    print("- These tests use simple policy gradient, not state-of-the-art algorithms")
    print("- Performance with PPO/SAC may differ significantly")
    print("- Tests verify basic learning capability, not optimal performance")
    print("- Pendulum is a simple environment; complex tasks need more validation")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = run_all_learning_tests()
    exit(0 if success else 1)

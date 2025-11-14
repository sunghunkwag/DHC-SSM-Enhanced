"""
Fixed PPO implementation with proper Gaussian policy for continuous actions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dhc_ssm.adapters.rl_policy_v2 import SimpleRLPolicy, SimpleRLValue


class GaussianPolicy(nn.Module):
    """Policy with Gaussian distribution for continuous actions."""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Initialize
        for m in [self.network, self.mean_layer]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.zeros_(layer.bias)
    
    def forward(self, obs):
        h = self.network(obs)
        mean = torch.tanh(self.mean_layer(h))  # Bound to [-1, 1]
        std = torch.exp(self.log_std.clamp(-20, 2))
        return mean, std
    
    def get_action(self, obs):
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob
    
    def evaluate_actions(self, obs, actions):
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
    
    def forward(self, obs):
        return self.network(obs)


def collect_rollouts(env, policy, value_fn, num_steps=2048):
    observations = []
    actions = []
    log_probs = []
    rewards = []
    dones = []
    values = []
    episode_rewards = []
    
    obs, _ = env.reset()
    episode_reward = 0
    
    for _ in range(num_steps):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob = policy.get_action(obs_tensor)
            value = value_fn(obs_tensor)
        
        action_np = action.squeeze(0).numpy()
        
        observations.append(obs)
        actions.append(action_np)
        log_probs.append(log_prob.item())
        values.append(value.item())
        
        next_obs, reward, terminated, truncated, _ = env.step(action_np)
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
    
    return (np.array(observations), np.array(actions), np.array(log_probs),
            np.array(rewards), np.array(dones), np.array(values), episode_rewards)


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = np.zeros_like(rewards)
    last_advantage = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_advantage = delta + gamma * lam * (1 - dones[t]) * last_advantage
    
    returns = advantages + values
    return advantages, returns


def ppo_update(policy, value_fn, policy_opt, value_opt, observations, actions,
               old_log_probs, advantages, returns, epochs=10, clip_eps=0.2, batch_size=64):
    
    obs_tensor = torch.FloatTensor(observations)
    actions_tensor = torch.FloatTensor(actions)
    old_log_probs_tensor = torch.FloatTensor(old_log_probs)
    advantages_tensor = torch.FloatTensor(advantages)
    returns_tensor = torch.FloatTensor(returns)
    
    # Normalize advantages
    advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
    
    dataset_size = len(observations)
    indices = np.arange(dataset_size)
    
    policy_losses = []
    value_losses = []
    
    for _ in range(epochs):
        np.random.shuffle(indices)
        
        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            batch_obs = obs_tensor[batch_indices]
            batch_actions = actions_tensor[batch_indices]
            batch_old_log_probs = old_log_probs_tensor[batch_indices]
            batch_advantages = advantages_tensor[batch_indices]
            batch_returns = returns_tensor[batch_indices]
            
            # Policy loss
            new_log_probs, entropy = policy.evaluate_actions(batch_obs, batch_actions)
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()
            
            policy_opt.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            policy_opt.step()
            
            # Value loss
            pred_values = value_fn(batch_obs).squeeze()
            value_loss = F.mse_loss(pred_values, batch_returns)
            
            value_opt.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(value_fn.parameters(), 0.5)
            value_opt.step()
            
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
    
    return np.mean(policy_losses), np.mean(value_losses)


def train_ppo(env_name="Pendulum-v1", num_episodes=100, seed=42):
    env = gym.make(env_name)
    env.reset(seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    policy = GaussianPolicy(obs_dim, action_dim, hidden_dim=64)
    value_fn = ValueNetwork(obs_dim, hidden_dim=64)
    
    policy_opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    value_opt = torch.optim.Adam(value_fn.parameters(), lr=1e-3)
    
    all_episode_rewards = []
    
    print(f"Training PPO on {env_name} for {num_episodes} episodes")
    
    while len(all_episode_rewards) < num_episodes:
        obs, actions, log_probs, rewards, dones, values, ep_rewards = collect_rollouts(
            env, policy, value_fn, num_steps=2048
        )
        
        all_episode_rewards.extend(ep_rewards)
        
        advantages, returns = compute_gae(rewards, values, dones)
        
        policy_loss, value_loss = ppo_update(
            policy, value_fn, policy_opt, value_opt,
            obs, actions, log_probs, advantages, returns
        )
        
        if len(all_episode_rewards) % 20 == 0:
            recent = np.mean(all_episode_rewards[-20:])
            print(f"  Episode {len(all_episode_rewards):3d}: reward={recent:8.2f}")
    
    env.close()
    
    initial = np.mean(all_episode_rewards[:10])
    final = np.mean(all_episode_rewards[-10:])
    
    return all_episode_rewards, initial, final


if __name__ == "__main__":
    print("="*70)
    print("FIXED PPO VALIDATION")
    print("="*70)
    print()
    
    rewards, initial, final = train_ppo("Pendulum-v1", num_episodes=100, seed=42)
    improvement = final - initial
    
    print(f"\nResults: {initial:.2f} -> {final:.2f} (improvement: {improvement:+.2f})")
    
    if improvement > 100:
        print("VALIDATION PASSED: Model learns successfully")
    else:
        print("VALIDATION FAILED: Insufficient learning")

"""Generate plots for fixed benchmark results."""

import json
import csv
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_learning_curves(csv_file, title, output_file):
    episodes, ssm_rewards, mlp_rewards = [], [], []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row['episode']))
            ssm_rewards.append(float(row['ssm_reward']))
            mlp_rewards.append(float(row['mlp_reward']))
    
    window = 10
    ssm_ma = np.convolve(ssm_rewards, np.ones(window)/window, mode='valid')
    mlp_ma = np.convolve(mlp_rewards, np.ones(window)/window, mode='valid')
    episodes_ma = episodes[window-1:]
    
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, ssm_rewards, alpha=0.3, color='blue', label='SSM (raw)')
    plt.plot(episodes, mlp_rewards, alpha=0.3, color='orange', label='MLP (raw)')
    plt.plot(episodes_ma, ssm_ma, color='blue', linewidth=2, label='SSM (MA-10)')
    plt.plot(episodes_ma, mlp_ma, color='orange', linewidth=2, label='MLP (MA-10)')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved: {output_file}")


def plot_reproducibility(json_file, output_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    runs = data['runs']
    
    plt.figure(figsize=(10, 6))
    
    for run in runs:
        seed = run['seed']
        rewards = run['rewards']
        episodes = list(range(1, len(rewards) + 1))
        
        window = 10
        ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        episodes_ma = episodes[window-1:]
        
        plt.plot(episodes, rewards, alpha=0.2)
        plt.plot(episodes_ma, ma, linewidth=2, label=f'Seed {seed}')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reproducibility Test: SSM across 3 seeds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved: {output_file}")


def generate_all_plots():
    print("Generating plots for fixed benchmarks...")
    
    os.makedirs("benchmarks_fixed/plots", exist_ok=True)
    
    if os.path.exists("benchmarks_fixed/pendulum_200.csv"):
        plot_learning_curves(
            "benchmarks_fixed/pendulum_200.csv",
            "Pendulum-v1 Learning Curves (Fixed PPO, 200 episodes)",
            "benchmarks_fixed/plots/pendulum_learning_curves.png"
        )
    
    if os.path.exists("benchmarks_fixed/hopper_200.csv"):
        plot_learning_curves(
            "benchmarks_fixed/hopper_200.csv",
            "Hopper-v4 Learning Curves (Fixed PPO, 200 episodes)",
            "benchmarks_fixed/plots/hopper_learning_curves.png"
        )
    
    if os.path.exists("benchmarks_fixed/reproducibility.json"):
        plot_reproducibility(
            "benchmarks_fixed/reproducibility.json",
            "benchmarks_fixed/plots/reproducibility.png"
        )
    
    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    generate_all_plots()

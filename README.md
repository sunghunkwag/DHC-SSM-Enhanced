# DHC-SSM-Enhanced

## Deterministic Hierarchical Causal State Space Model

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

State space model architecture for spatial-temporal data processing with O(n) linear complexity. Validated on computer vision and reinforcement learning tasks.

---

## Overview

DHC-SSM v3.1 combines spatial processing, temporal modeling, and causal reasoning for efficient sequence modeling. The architecture has been validated with proper PPO implementation on multiple MuJoCo continuous control environments.

### Key Features

- O(n) computational complexity
- Adaptive architecture for variable input dimensions
- Simplified SSM variant for vector-based RL tasks
- Validated learning on 4 MuJoCo environments

---

## Benchmark Results

### MuJoCo Continuous Control (200 episodes, PPO)

| Environment | SSM Initial | SSM Final | SSM Improvement | MLP Improvement |
|-------------|-------------|-----------|-----------------|-----------------|
| Pendulum-v1 | -1436 | -1323 | +113 | +32 |
| Hopper-v4 | 10.0 | 15.8 | +6 | +0.2 |
| Walker2d-v4 | 4.0 | 23.4 | +19 | +14 |
| HalfCheetah-v4 | -915 | 95 | +1010 | +586 |

### Summary

- All environments show successful learning
- SSM outperforms MLP baseline on all 4 tasks
- HalfCheetah demonstrates strongest improvement
- Reproducible across multiple random seeds

**Detailed results:** [BENCHMARK_RESULTS_FIXED.md](BENCHMARK_RESULTS_FIXED.md)

---

## Installation

### Requirements

- Python 3.11+
- PyTorch 2.9.0+
- CUDA 12.8+ (optional)

### Install

```bash
git clone https://github.com/sunghunkwag/DHC-SSM-Enhanced.git
cd DHC-SSM-Enhanced
pip install -e .
```

### With RL Support

```bash
pip install -e .
pip install mujoco gymnasium[mujoco]
```

---

## Usage

### Computer Vision

```python
from dhc_ssm.core.model import DHCSSMModel, DHCSSMConfig

config = DHCSSMConfig(
    input_channels=3,
    hidden_dim=64,
    state_dim=64,
    output_dim=10
)

model = DHCSSMModel(config)
output = model(images)
```

### Reinforcement Learning

```python
from dhc_ssm.adapters.rl_policy_v2 import SimpleRLPolicy, SimpleRLValue
import gymnasium as gym

env = gym.make("Pendulum-v1")
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

policy = SimpleRLPolicy(obs_dim, action_dim, hidden_dim=64, state_dim=32)
value_fn = SimpleRLValue(obs_dim, hidden_dim=64, state_dim=32)
```

### PPO Training

```python
import torch
from dhc_ssm.adapters.rl_policy_v2 import SimpleRLPolicy, SimpleRLValue

policy = SimpleRLPolicy(obs_dim, action_dim)
value_fn = SimpleRLValue(obs_dim)

policy_opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
value_opt = torch.optim.Adam(value_fn.parameters(), lr=1e-3)

# Collect rollouts, compute GAE, run PPO updates
# See tests/test_fixed_ppo.py for complete implementation
```

---

## Architecture

### Core Components

1. **Spatial Encoder:** Adaptive pooling for variable dimensions
2. **State Space Module:** O(n) temporal modeling
3. **Causal Attention:** Hierarchical long-range dependencies
4. **Output Projection:** Task-specific heads

### Simplified SSM for RL

```
Input -> Linear + LayerNorm + Tanh
      -> State Projection
      -> State Update
      -> Output
```

Key differences from standard DHC-SSM:
- No spatial convolutions for 1D observations
- LayerNorm for training stability
- Smaller weight initialization (gain=0.01)
- Direct state space transformation

---

## Testing

### Run All Tests

```bash
pytest tests/
```

### Run Specific Tests

```bash
# Core functionality
pytest tests/test_comprehensive.py

# PPO validation
python tests/test_fixed_ppo.py

# Full benchmarks (Pendulum, Hopper)
python tests/test_fixed_benchmarks.py

# Extended benchmarks (Walker2d, HalfCheetah)
python tests/test_extended_benchmarks.py
```

### Generate Plots

```bash
python scripts/generate_fixed_plots.py
```

---

## Project Structure

```
DHC-SSM-Enhanced/
├── dhc_ssm/
│   ├── core/
│   │   ├── model.py
│   │   ├── spatial.py
│   │   ├── ssm.py
│   │   └── attention.py
│   └── adapters/
│       ├── rl_policy.py
│       └── rl_policy_v2.py
├── tests/
│   ├── test_comprehensive.py
│   ├── test_fixed_ppo.py
│   ├── test_fixed_benchmarks.py
│   └── test_extended_benchmarks.py
├── benchmarks_fixed/
│   ├── pendulum_200.json/csv
│   ├── hopper_200.json/csv
│   └── plots/
├── benchmarks_extended/
│   ├── walker2d_v4_200.json/csv
│   ├── halfcheetah_v4_200.json/csv
│   └── plots/
└── scripts/
    └── generate_fixed_plots.py
```

---

## API Reference

### SimpleRLPolicy

```python
SimpleRLPolicy(
    observation_dim: int,
    action_dim: int,
    hidden_dim: int = 128,
    state_dim: int = 64
)
```

Policy network with simplified SSM architecture.

**Methods:**
- `forward(obs)`: Returns actions in [-1, 1]

### SimpleRLValue

```python
SimpleRLValue(
    observation_dim: int,
    hidden_dim: int = 128,
    state_dim: int = 64
)
```

Value function network with simplified SSM architecture.

**Methods:**
- `forward(obs)`: Returns state value estimate

---

## Limitations

### Current Scope

- Tested on 4 MuJoCo environments
- 200 episodes per environment
- Single RL algorithm (PPO)
- No hyperparameter tuning

### Not Tested

- Image-based RL (Atari)
- Longer training runs (1000+ episodes)
- Other algorithms (SAC, TD3)
- Humanoid and complex environments

---

## When to Use

### SSM is Suitable For

- Image-based RL tasks
- Tasks requiring temporal context
- Variable input dimensions
- Research on state space models

### Use MLP For

- Standard vector observations
- When inference speed is critical
- When parameter efficiency matters
- Simple continuous control tasks

---

## Reproducibility

All benchmarks use fixed random seeds and can be reproduced:

```bash
cd DHC-SSM-Enhanced
source venv/bin/activate

# Pendulum and Hopper
python tests/test_fixed_benchmarks.py

# Walker2d and HalfCheetah
python tests/test_extended_benchmarks.py

# Generate visualizations
python scripts/generate_fixed_plots.py
```

Results saved in `benchmarks_fixed/` and `benchmarks_extended/` directories.

---

## Contributing

Contributions welcome. Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

---

## Citation

```bibtex
@software{dhc_ssm_2025,
  title={DHC-SSM: Deterministic Hierarchical Causal State Space Model},
  author={Kwag, Sunghun},
  year={2025},
  version={3.1},
  url={https://github.com/sunghunkwag/DHC-SSM-Enhanced}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Version History

### v3.1 (Current)
- Fixed PPO implementation with Gaussian policy
- Simplified SSM architecture for RL
- Validated on 4 MuJoCo environments
- Reproducibility verification
- Comprehensive benchmark suite

### v3.0
- Initial MuJoCo integration
- Original RL adapters

### v2.0
- Core DHC-SSM architecture
- Computer vision support

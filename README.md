# DHC-SSM-Enhanced

> **Research Note**: This is an experimental implementation used to probe the limits of deterministic state space models. It is **not** production-ready software. The O(n) complexity claim relies on specific linearity assumptions that may not hold in complex causal hierarchies.

## Deterministic Hierarchical Causal State Space Model (Experimental)

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An attempt to implement a state space model architecture for spatial-temporal data processing with O(n) linear complexity. The primary goal is to test if such architectures can effectively model causal dependencies without the quadratic cost of Transformers.

---

## Overview

DHC-SSM v3.1 combines spatial processing, temporal modeling, and causal reasoning. This repository serves as a testbed for validating these concepts on standard continuous control benchmarks (MuJoCo).

**Hypothesis**: Can a simplified SSM variant with O(n) complexity match the performance of standard MLPs on vector-based RL tasks?

### Experimental Capabilities

- **O(n) computational path**: Theoretical linear scaling (subject to implementation overhead).
- **Adaptive architecture**: Handles variable input dimensions via pooling (experimental).
- **Simplified SSM variant**: A stripped-down version specifically for vector-based RL.

---

## Experimental Observations

> **Note**: Positive results here indicate that the model *can* learn, not that it is superior to established baselines in all contexts.

### MuJoCo Continuous Control (200 episodes, PPO, fixed seed)

| Environment | SSM Initial | SSM Final | Change | MLP Change (Baseline) |
|-------------|-------------|-----------|--------|-----------------------|
| Pendulum-v1 | -1436 | -1323 | +113 | +32 |
| Hopper-v4 | 10.0 | 15.8 | +6 | +0.2 |
| Walker2d-v4 | 4.0 | 23.4 | +19 | +14 |
| HalfCheetah-v4 | -915 | 95 | +1010 | +586 |

**Observation**: In this specific hyperparameter regime, the SSM variant showed successful learning curves and, in some runs (e.g., HalfCheetah), achieved higher final rewards than the MLP baseline. *This does not imply general superiority.*

**Detailed logs**: [BENCHMARK_RESULTS_FIXED.md](BENCHMARK_RESULTS_FIXED.md)

---

## Failure Analysis & Limitations

This section documents where the model breaks or behaves unexpectedly.

1.  **Numerical Instability**: In deeper hierarchies (>4 layers), gradients tend to vanish or explode despite normalization. The O(n) recurrence seems less robust to deep signal propagation than attention mechanisms.
2.  **Causal Leakage**: The "causal" masking in the spatial encoder is approximate. Strictly speaking, some future information might leak through padding artifacts in convolution operations.
3.  **Initialization Sensitivity**: The model is highly sensitive to the initialization variance of the state projection matrices. Incorrect scaling leads to immediate divergence.
4.  **Limited Scope**: Validated only on low-dimensional vector observations. High-dimensional image-based RL remains unverified and likely problematic due to the "Causal Leakage" issue mentioned above.

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

## Usage (Experimental)

### Computer Vision (Unverified)

```python
from dhc_ssm.core.model import DHCSSMModel, DHCSSMConfig

config = DHCSSMConfig(
    input_channels=3,
    hidden_dim=64,
    state_dim=64,
    output_dim=10
)

# Warning: Stability issues observed with hidden_dim > 128
model = DHCSSMModel(config)
output = model(images)
```

### Reinforcement Learning (Tested)

```python
from dhc_ssm.adapters.rl_policy_v2 import SimpleRLPolicy, SimpleRLValue
import gymnasium as gym

env = gym.make("Pendulum-v1")
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

policy = SimpleRLPolicy(obs_dim, action_dim, hidden_dim=64, state_dim=32)
value_fn = SimpleRLValue(obs_dim, hidden_dim=64, state_dim=32)
```

---

## Reproduction

Benchmarks use fixed random seeds for reproducibility.

```bash
python tests/test_fixed_benchmarks.py
python tests/test_extended_benchmarks.py
```

---

## Citation

If you use this code for research (e.g., as a baseline for failure analysis), please cite:

```bibtex
@software{dhc_ssm_2025,
  title={DHC-SSM: Deterministic Hierarchical Causal State Space Model (Experimental)},
  author={Kwag, Sunghun},
  year={2025},
  version={3.1},
  url={https://github.com/sunghunkwag/DHC-SSM-Enhanced}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

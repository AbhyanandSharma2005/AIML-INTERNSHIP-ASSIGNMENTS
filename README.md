# PPO CartPole-v1 Reinforcement Learning

**Submitted By:** Lakshay Saini
**Registration No:** 23BAI10890

This project implements the Proximal Policy Optimization (PPO) algorithm to solve the Gymnasium CartPole-v1 environment using Stable-Baselines3.

---

## Project Structure
* `train.py` - Script to initialize the environment, train the PPO agent, and save training logs.
* `evaluate.py` - Evaluates the trained model over 100 test episodes to log deterministic performance metrics.
* `test.py` - Renders a live visual window showing the trained model balancing the pole.
* `ppo_model.zip` - Saved policy network weights.
* `learning_curve.png` - Training performance graph showing reward progression.
* `evaluation_report.txt` - Text document tracking final evaluation results.

---

## Installation & Setup

1. **Create and activate a clean Conda environment:**
   ```bash
   conda create -n my_env python=3.10 -y
   conda activate my_env

```

2. **Install the required packages:**
*(Note: Quotes around `stable-baselines3[extra]` are required for macOS Zsh compatibility to avoid pattern matching errors).*
```bash
pip install "stable-baselines3[extra]" gymnasium matplotlib

```

---

## How to Run

Execute the scripts in the following order to generate your deliverables:

### 1. Train the Agent

Runs the PPO agent for 50,000 timesteps, saves the model checkpoint, and generates the training progress chart.

```bash
python train.py

```

### 2. Evaluate Performance

Tests the saved checkpoint over 100 separate episodes and writes the performance numbers into a clean text report.

```bash
python evaluate.py

```

### 3. Visual Testing

Launches a live visual rendering window on your Mac screen to watch the trained model in action.

```bash
python test.py

```

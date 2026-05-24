

https://github.com/user-attachments/assets/74f8ecc0-320c-4844-ba58-0b7df863add6

# Reinforcement Learning Control of a Tendon-Driven Continuum Robot

Reinforcement learning agent that learns to position the tip of a soft, tendon-driven continuum robot at arbitrary 3D targets within its workspace. Built on PyElastica (Cosserat rod simulation) and Stable-Baselines3 PPO, with behavioural cloning warm-start from a PID controller. The trained agent surpasses the PID baseline.

---

## Project Overview

This project explores whether modern deep RL can learn to control a high-dimensional, nonlinear soft robotic system better than a hand-tuned classical controller.

The robot is an 8-tendon continuum manipulator simulated as a Cosserat rod. The agent observes the tip position, velocity, and tendon state, and must output 8 tendon tensions to drive the tip toward a randomly sampled 3D target position.

### What Was Done

The project went through several distinct phases:

**1. Environment construction**
A Gymnasium environment wraps a PyElastica simulation. The rod is actuated by 8 tendons (4 long, 4 short) routed through vertebrae along its length. Observations include tip position history, target, tip velocity, node speeds and positions, and recent action history (105 dimensions total). Actions are 8 normalized tendon tensions in `[-1, 1]`, scaled to physical units `[0, 8]N` inside the environment.

**2. Reward shaping**
A piecewise-continuous reward combines a quadratic bowl near the goal with a linear approach reward far from the goal, plus a stillness bonus for settling at target and a best-distance progress bonus. Several penalty terms (antagonist tendon activation, tension changes, node speeds) were prototyped and removed during development to keep the reward signal clean.

**3. PID baseline**
A position-based PID controller was implemented as a benchmark. It can reach most targets in 1-2 seconds but oscillates near difficult positions and cannot access the full workspace. Average episode reward: ~1442.

**4. Behavioural cloning warm-start**
Pure PPO from random initialization repeatedly converged to degenerate policies (rod thrashing, settling at origin, exploiting reward shaping). To break out of these local optima, 100 episodes of PID demonstrations (25,000 transitions) were collected and used to pretrain the policy network via supervised learning. The BC policy reaches near-zero validation loss but suffers from compounding error in closed-loop deployment.

**5. PPO fine-tuning with collapse termination**
PPO fine-tunes from the BC initialization. To prevent BC's covariate shift from cascading into rod-destabilizing states, episodes terminate early with a penalty when node speeds exceed a safe threshold. This focuses learning on the regime where smooth control is achievable.

**6. Result**
After ~5.5M training steps, the agent achieves an average episode reward of ~1483, surpassing the PID baseline (~1442). Episodes run to full length without collapse, the rod reliably reaches targets, and the agent generalizes across the workspace including regions PID could not reach.

### Known Behavior Caveats

The trained agent occasionally exploits the quadratic reward shape by oscillating around the target rather than fully settling. A planned mitigation (tip speed penalty gated to near-goal regions) is described in the development notes and can be applied via a short fine-tune run.

### Method                          | Avg Reward | Notes

PID (baseline)                  | ~1442      | Hand-tuned, can't reach all targets

PPO from scratch                | -1700      | Converged to degenerate policy

PPO + BC warm-start             | ~1483      | Surpasses PID, occasional oscillation

---

## Stack

- **Stable-Baselines3** — PPO implementation, VecNormalize observation/reward normalization, checkpointing
- **Gymnasium** — environment interface
- **PyElastica** — Cosserat rod physics simulation (open-source)
- **PyTorch** — neural network backend (via SB3)
- **Numba** — JIT compilation of reward and state-extraction hot paths
- **NumPy** — numerical core
- **MoviePy + Matplotlib** — visualization of episode rollouts

---

## Project Structure

```
rl_cr_robot/
├── main.py                       # entry point: train / resume / test
├── config.py                     # all hyperparameters (env, PPO, paths)
│
├── env/
│   ├── cr_env.py                 # Gymnasium environment wrapping PyElastica
│   └── reward.py                 # reward components (numba-compiled)
│
├── sim/
│   ├── rod_simulator.py          # PyElastica setup, callbacks, OctoTendonForces
│   └── sim_params.py             # physical constants (rod, tendon, sim)
│
├── training/
│   ├── train.py                  # PPO training loop, checkpoint resume
│   ├── test.py                   # evaluation rollouts with video output
│   ├── callbacks.py              # logging and checkpoint callbacks
│   ├── collect_pid_demos.py      # PID demonstration collector
│   ├── train_bc.py               # behavioural cloning pretraining
│   └── test_pid.py               # PID controller standalone tester
│
├── common/
│   ├── env_utils.py              # SubprocVecEnv factory
│   ├── utils.py                  # numba-compiled state extraction
│   └── visualizer.py             # video rendering of episodes
│
├── checkpoints/                  # PPO model checkpoints
├── logs/                         # Monitor CSVs per environment
├── ppo_tensorboard/              # TensorBoard event logs
└── test_runs/                    # rendered videos and per-step data
```

---

## Usage

All commands are run from the project root.

### Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.12+, ~2GB RAM for replay buffer, multi-core CPU recommended for parallel environments.

### Quick Start — Train From Scratch

```bash
python main.py --scratch
```

Trains a PPO agent from random initialization. Will likely converge to a degenerate policy without BC warm-start — see below for the recommended pipeline.

### Recommended Pipeline (BC + PPO)

The full pipeline that produces the working agent:

```bash
# 1. Collect PID demonstrations
python -m training.collect_pid_demos \
    --num-episodes 100 \
    --output training/demo_data/pid_demos.npz

# 2. Behavioural cloning pretraining
python -m training.train_bc \
    --demos training/demo_data/pid_demos.npz \
    --vecnorm training/demo_data/pid_demos_vecnormalize.pkl \
    --output checkpoints/ppo_robot_bc_init.zip \
    --epochs 50

# 3. PPO fine-tune from BC checkpoint
python main.py --checkpoint --checkpoint-path checkpoints/ppo_robot_bc_init.zip
```

### Resume From a Checkpoint

```bash
python main.py --checkpoint --checkpoint-path checkpoints/ppo_robot_<steps>_steps.zip
```

If `--checkpoint-path` is omitted, resumes from the latest checkpoint in the directory automatically.

### Evaluate a Trained Policy

```bash
python main.py --test --checkpoint-path checkpoints/ppo_robot_<steps>_steps.zip
```

Runs one episode with the loaded policy, prints per-step state and actions, saves a 3D animation to `test_runs/videos/`.

### Test the PID Baseline

```bash
python -m training.test_pid                              # random target
python -m training.test_pid --target 0.20 0.05 -0.05     # specific target
```

### Monitor Training

```bash
tensorboard --logdir ppo_tensorboard/
```

Key metrics to watch: `rollout/ep_rew_mean` (should climb), `train/explained_variance` (should reach 0.7+), `rollout/ep_len_mean` (should reach episode max as collapse rate drops).

---

## Configuration

All hyperparameters live in `config.py` as dataclasses:

- `EnvConfig` — workspace bounds, history lengths, simulation steps per RL step
- `PPOConfig` — network architecture, learning rate, n_steps, batch size, etc.
- `PathConfig` — checkpoint, log, and output directories

Physical parameters are in `sim/sim_params.py`:

- `RodParams` — geometry, density, Young's modulus, shear modulus
- `TendonParams` — tendon count, max tension, vertebra layout
- `SimParams` — timestep, episode duration, gravity, damping

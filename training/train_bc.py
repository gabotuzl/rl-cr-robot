"""
train_bc.py
-----------
Behavioural cloning pretraining. Loads PID demonstrations and trains the
PPO policy network's mean output to imitate the PID actions via MSE loss.

The trained weights are saved as a PPO checkpoint that the standard
training loop can resume from with --checkpoint.

Usage (from project root):
    python -m training.train_bc \
        --demos training/pid_demos.npz \
        --vecnorm training/pid_demos_vecnormalize.pkl \
        --output checkpoints/ppo_robot_bc_init.zip \
        --epochs 50
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.cr_env import cr_env
from config import CONFIG


def train_behavioural_cloning(demos_path: str,
                              vecnorm_path: str,
                              output_path: str,
                              epochs: int = 50,
                              batch_size: int = 256,
                              learning_rate: float = 3e-4,
                              val_split: float = 0.1):

    # ---- Load demonstrations ----
    print(f"Loading demos from {demos_path}...")
    data = np.load(demos_path)
    observations = data['observations']
    actions = data['actions']
    print(f"  observations: {observations.shape}")
    print(f"  actions:      {actions.shape}")
    print(f"  obs range:    [{observations.min():.3f}, {observations.max():.3f}]")
    print(f"  action range: [{actions.min():.3f}, {actions.max():.3f}]")

    # ---- Build environment to instantiate PPO with correct spaces ----
    print("\nBuilding environment to instantiate PPO model...")
    env = DummyVecEnv([lambda: cr_env()])
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False  # don't update stats during BC

    # ---- Create PPO model with the architecture used for fine-tuning ----
    policy_kwargs = dict(
        net_arch=dict(pi=CONFIG.ppo.net_arch_pi, vf=CONFIG.ppo.net_arch_vf),
        ortho_init=True,
    )
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        n_steps=CONFIG.ppo.n_steps,
        batch_size=CONFIG.ppo.batch_size,
        gae_lambda=CONFIG.ppo.gae_lambda,
        gamma=CONFIG.ppo.gamma,
        n_epochs=CONFIG.ppo.n_epochs,
        learning_rate=CONFIG.ppo.learning_rate,
        clip_range=CONFIG.ppo.clip_range,
        ent_coef=CONFIG.ppo.ent_coef,
        vf_coef=CONFIG.ppo.vf_coef,
        max_grad_norm=CONFIG.ppo.max_grad_norm,
        verbose=0,
    )

    device = model.policy.device
    print(f"Using device: {device}")

    # ---- Prepare PyTorch dataset ----
    obs_tensor = torch.tensor(observations, dtype=torch.float32)
    act_tensor = torch.tensor(actions, dtype=torch.float32)

    # Train/val split
    n_total = len(obs_tensor)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    perm = torch.randperm(n_total)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    train_ds = TensorDataset(obs_tensor[train_idx], act_tensor[train_idx])
    val_ds = TensorDataset(obs_tensor[val_idx], act_tensor[val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"\nTrain set: {n_train} | Val set: {n_val}")

    # ---- Training loop ----
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')

    print(f"\nStarting BC training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.policy.train()
        train_loss = 0.0
        n_train_batches = 0

        for obs_batch, act_batch in train_loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)

            # Get action distribution from policy and use its mean as the prediction
            # (this is what deterministic=True would produce at inference)
            features = model.policy.extract_features(obs_batch)
            if model.policy.share_features_extractor:
                latent_pi, _ = model.policy.mlp_extractor(features)
            else:
                pi_features = features[0] if isinstance(features, tuple) else features
                latent_pi = model.policy.mlp_extractor.forward_actor(pi_features)
            predicted_actions = model.policy.action_net(latent_pi)

            loss = loss_fn(predicted_actions, act_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), max_norm=0.5)
            optimizer.step()

            train_loss += loss.item()
            n_train_batches += 1

        train_loss /= n_train_batches

        # ---- Validate ----
        model.policy.eval()
        val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for obs_batch, act_batch in val_loader:
                obs_batch = obs_batch.to(device)
                act_batch = act_batch.to(device)

                features = model.policy.extract_features(obs_batch)
                if model.policy.share_features_extractor:
                    latent_pi, _ = model.policy.mlp_extractor(features)
                else:
                    pi_features = features[0] if isinstance(features, tuple) else features
                    latent_pi = model.policy.mlp_extractor.forward_actor(pi_features)
                predicted_actions = model.policy.action_net(latent_pi)

                val_loss += loss_fn(predicted_actions, act_batch).item()
                n_val_batches += 1

        val_loss /= n_val_batches

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss

        marker = "  ← best" if improved else ""
        print(f"Epoch {epoch:>3}/{epochs} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f}{marker}")

    # ---- Save model and VecNormalize together ----
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    model.save(output_path)

    vecnorm_out = output_path.replace(".zip", ".pkl").replace(
        "ppo_robot", "ppo_robot_vecnormalize"
    )
    # Re-enable training so PPO can update stats during fine-tuning
    env.training = True
    env.save(vecnorm_out)

    print(f"\n--- BC Training Complete ---")
    print(f"Best val loss:    {best_val_loss:.5f}")
    print(f"Saved model to:   {output_path}")
    print(f"Saved vecnorm to: {vecnorm_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos",         type=str, required=True)
    parser.add_argument("--vecnorm",       type=str, required=True)
    parser.add_argument("--output",        type=str, default="checkpoints/ppo_robot_bc_init.zip")
    parser.add_argument("--epochs",        type=int, default=50)
    parser.add_argument("--batch-size",    type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--val-split",     type=float, default=0.1)
    args = parser.parse_args()

    train_behavioural_cloning(
        demos_path=args.demos,
        vecnorm_path=args.vecnorm,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_split=args.val_split,
    )
"""
collect_pid_demos.py
--------------------
Collects PID controller demonstrations through the same VecNormalize wrapper
that the RL agent will use, so observations are normalized consistently.

Output: .npz file with normalized observations + RL-space actions.

Tendon vector layout (8 elements):
    Index   Tendon              Active when
    -----   -----------------   -------------
    0       VLT (+Z)            target z > 0
    1       HLT (+Y)            target y > 0
    2       VLT (-Z)            target z < 0
    3       HLT (-Y)            target y < 0
    4       VST (+Z)            target z < 0  (antagonist of VLT(-Z))
    5       HST (+Y)            target y < 0  (antagonist of HLT(-Y))
    6       VST (-Z)            target z > 0  (antagonist of VLT(+Z))
    7       HST (-Y)            target y > 0  (antagonist of HLT(+Y))

Usage:
    python collect_pid_demos.py --num-episodes 100 --output pid_demos.npz
"""

import argparse
import os
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tqdm import tqdm, trange

from env.cr_env import cr_env
from sim.sim_params import TENDON_PARAMS
from config import CONFIG


# ---------------------------------------------------------------------------
# PID controller — mirrors the original logic
# ---------------------------------------------------------------------------

class PIDController:
    """
    PID controller for the 8-tendon continuum robot.

    Maintains 4 tension magnitudes (VLT, HLT, VST, HST) and decides which
    of the 8 physical tendons gets each magnitude based on target sign.
    """

    def __init__(self, kp_long: float = 1.0, kp_short: float = 0.5):
        self.kp_long = kp_long
        self.kp_short = kp_short
        self.reset()

    def reset(self):
        self.tension_vertical_long = 0.0
        self.tension_horizontal_long = 0.0
        self.tension_vertical_short = 0.0
        self.tension_horizontal_short = 0.0

    @staticmethod
    def positive_or_zero(value: float) -> float:
        return max(0.0, value)

    def compute(self, current_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        """
        Returns 8-element tension vector in physical units [0, max_tension].
        """
        x_des, y_des, z_des = target_pos
        x_cur, y_cur, z_cur = current_pos

        z_sign = np.sign(z_des) if z_des != 0 else 1.0
        y_sign = np.sign(y_des) if y_des != 0 else 1.0

        dx = x_cur - x_des
        dy = (y_cur - y_des) * y_sign
        dz = (z_cur - z_des) * z_sign

        self.tension_vertical_long    = self.positive_or_zero(self.tension_vertical_long    - dz * self.kp_long)
        self.tension_horizontal_long  = self.positive_or_zero(self.tension_horizontal_long  - dy * self.kp_long)
        self.tension_vertical_short   = self.positive_or_zero(self.tension_vertical_short   + (dx + dz) * self.kp_short)
        self.tension_horizontal_short = self.positive_or_zero(self.tension_horizontal_short + (dx + dy) * self.kp_short)

        # Build 8-element tension vector based on target signs
        # Layout: [VLT(+Z), HLT(+Y), VLT(-Z), HLT(-Y), VST(+Z), HST(+Y), VST(-Z), HST(-Y)]
        tensions = np.zeros(8, dtype=np.float64)

        if z_sign > 0:
            tensions[0] = self.tension_vertical_long      # VLT(+Z)
            tensions[6] = self.tension_vertical_short     # VST(-Z) antagonist
        else:
            tensions[2] = self.tension_vertical_long      # VLT(-Z)
            tensions[4] = self.tension_vertical_short     # VST(+Z) antagonist

        if y_sign < 0:
            tensions[1] = self.tension_horizontal_long    # HLT(+Y)
            tensions[7] = self.tension_horizontal_short   # HST(-Y) antagonist
        else:
            tensions[3] = self.tension_horizontal_long    # HLT(-Y)
            tensions[5] = self.tension_horizontal_short   # HST(+Y) antagonist

        return tensions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def physical_to_rl_action(tensions: np.ndarray, max_tension: float) -> np.ndarray:
    """Convert physical tendon tensions [0, max] to RL action space [-1, 1]."""
    return (tensions / max_tension) * 2.0 - 1.0


def warmup_vecnormalize(env: VecNormalize, num_steps: int = 5000):
    print(f"Warming up VecNormalize for {num_steps} steps with random actions...")
    obs = env.reset()
    for _ in tqdm(range(num_steps), desc="Warmup"):
        action = env.action_space.sample().reshape(1, -1)
        obs, _, done, _ = env.step(action)
        if done.any():
            obs = env.reset()
    print("Warmup complete.")

# ---------------------------------------------------------------------------
# Main collection routine
# ---------------------------------------------------------------------------

def collect_demonstrations(num_episodes: int, output_path: str,
                            kp_long: float = 1.0, kp_short: float = 0.5,
                            warmup_steps: int = 5000):

    env = DummyVecEnv([lambda: cr_env()])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    warmup_vecnormalize(env, num_steps=warmup_steps)

    pid = PIDController(kp_long=kp_long, kp_short=kp_short)
    max_tension = TENDON_PARAMS.max_tension

    all_obs, all_actions = [], []
    episode_rewards = []

    for ep in trange(num_episodes, desc="Collecting demos"):
        obs = env.reset()
        target_pos = np.array(env.get_attr("target_position")[0])
        pid.reset()

        done = False
        ep_reward = 0.0
        ep_transitions = 0

        while not done:
            current_pos = np.array(env.get_attr("state")[0])

            pid_tensions = pid.compute(current_pos, target_pos)
            pid_tensions = np.clip(pid_tensions, 0.0, max_tension)

            rl_action = physical_to_rl_action(pid_tensions, max_tension)
            rl_action_batched = rl_action.reshape(1, -1)

            all_obs.append(obs[0].copy())
            all_actions.append(rl_action.copy())

            obs, reward, done_arr, _ = env.step(rl_action_batched)
            done = bool(done_arr[0])
            ep_reward += float(reward[0])
            ep_transitions += 1

        episode_rewards.append(ep_reward)
        print(f"Episode {ep + 1}/{num_episodes}: "
              f"reward={ep_reward:.2f}, steps={ep_transitions}, target={target_pos}")

    obs_array = np.array(all_obs)
    act_array = np.array(all_actions)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez(output_path,
             observations=obs_array,
             actions=act_array,
             episode_rewards=np.array(episode_rewards))

    vecnorm_path = output_path.replace(".npz", "_vecnormalize.pkl")
    env.save(vecnorm_path)

    print(f"\n--- Collection Summary ---")
    print(f"Transitions:     {len(all_obs):>8,}")
    print(f"Episodes:        {num_episodes:>8}")
    print(f"Avg reward:      {np.mean(episode_rewards):>8.2f}")
    print(f"Action range:    [{act_array.min():.3f}, {act_array.max():.3f}]")
    print(f"Action std:      {act_array.std():.3f}")
    print(f"Obs shape:       {obs_array.shape}")
    print(f"Saved demos to:  {output_path}")
    print(f"Saved vecnorm:   {vecnorm_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--output",       type=str, default="pid_demos.npz")
    parser.add_argument("--kp-long",      type=float, default=1.0)
    parser.add_argument("--kp-short",     type=float, default=2.0)
    parser.add_argument("--warmup-steps", type=int, default=5000)
    args = parser.parse_args()

    collect_demonstrations(
        num_episodes=args.num_episodes,
        output_path=args.output,
        kp_long=args.kp_long,
        kp_short=args.kp_short,
        warmup_steps=args.warmup_steps,
    )
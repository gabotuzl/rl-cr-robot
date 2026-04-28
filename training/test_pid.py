"""
test_pid.py
-----------
Quick tester for the PID controller. Runs a single episode with a fixed
target, prints per-step state, and saves a video of the rod's motion.

Usage (from project root):
    python -m training.test_pid
    python -m training.test_pid --target 0.20 0.05 -0.05
"""

import argparse
import os
import numpy as np

from env.cr_env import cr_env
from sim.sim_params import TENDON_PARAMS
from common.visualizer import render_episode
from training.collect_pid_demos import PIDController, physical_to_rl_action
from config import CONFIG


def run_pid_episode(target_pos: np.ndarray = None,
                    kp_long: float = 1.0,
                    kp_short: float = 0.5,
                    output_video: str = "test_runs/videos/pid_test.mp4",
                    verbose: bool = True):

    env = cr_env()
    obs, _ = env.reset()

    # Override target if one was provided
    if target_pos is not None:
        env.target_position = np.array(target_pos)
        env.delta = env.target_position - env.state
        env.best_distance = np.linalg.norm(env.delta)

    target_pos = np.array(env.target_position)
    print(f"\nTarget position: {target_pos}")
    print(f"Starting tip position: {env.state}\n")

    pid = PIDController(kp_long=kp_long, kp_short=kp_short)
    max_tension = TENDON_PARAMS.max_tension

    done = False
    step = 0
    episode_reward = 0.0
    min_dist = np.inf
    min_dist_step = 0

    while not done:
        current_pos = np.array(env.state)
        pid_tensions = pid.compute(current_pos, target_pos)
        pid_tensions = np.clip(pid_tensions, 0.0, max_tension)
        rl_action = physical_to_rl_action(pid_tensions, max_tension)

        obs, reward, terminated, truncated, info = env.step(rl_action)
        done = terminated or truncated
        step += 1
        episode_reward += float(reward)

        dist = np.linalg.norm(env.target_position - env.state)
############33
        x_des, y_des, z_des = env.target_position
        x_cur, y_cur, z_cur = env.state

        z_sign = np.sign(z_des) if z_des != 0 else 1.0
        y_sign = np.sign(y_des) if y_des != 0 else 1.0

        dx = x_cur - x_des
        dy = (y_cur - y_des) * y_sign
        dz = (z_cur - z_des) * z_sign

##############
        if dist < min_dist:
            min_dist = dist
            min_dist_step = step

        if verbose and step % 10 == 0:
            print(f"Step {step:>3} | dist={dist:.4f} | "
                  f"tip={np.round(env.state, 3)} | "
                  f"tensions={np.round(pid_tensions, 2)}")
            print(f"dx: {dx}\tdy:{dy}\tdz:{dz}")

    # Final summary
    print(f"\n--- PID Episode Summary ---")
    print(f"Total steps:       {step}")
    print(f"Total reward:      {episode_reward:.2f}")
    print(f"Final tip pos:     {np.round(env.state, 4)}")
    print(f"Final dist:        {np.linalg.norm(env.target_position - env.state):.4f}")
    print(f"Min dist achieved: {min_dist:.4f} at step {min_dist_step}")

    # Render video
    rod_data = env.callback_data_rod_object.copy()
    render_episode(
        position_data=rod_data['position'],
        target_position=target_pos,
        output_path=output_video,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",   nargs=3, type=float, default=None,
                        help="Target X Y Z (e.g. --target 0.2 0.05 -0.05)")
    parser.add_argument("--kp-long",  type=float, default=0.8)
    parser.add_argument("--kp-short", type=float, default=1.4)
    parser.add_argument("--output",   type=str, default="test_runs/videos/pid_test.mp4")
    args = parser.parse_args()

    run_pid_episode(
        target_pos=args.target,
        kp_long=args.kp_long,
        kp_short=args.kp_short,
        output_video=args.output,
    )
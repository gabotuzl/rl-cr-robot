import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.cr_env import cr_env
from common.visualizer import render_episode
from config import CONFIG
from sim.sim_params import TENDON_PARAMS


def run_testing(checkpoint_path: str = None, num_episodes: int = 1):
    # Deriving vecnorm path automatically from checkpoint path

    env = DummyVecEnv([lambda: cr_env()])

    if checkpoint_path is None:
        # Fresh untrained model — random weights
        print("No checkpoint provided — running with fresh untrained policy...")
        env = VecNormalize(env, norm_obs=True, norm_reward=False)
        model = PPO("MlpPolicy", env, ent_coef=CONFIG.ppo.ent_coef, verbose=0)
        deterministic = False  # sample from distribution to see exploration behaviour
    else:
        vecnorm_path = checkpoint_path.replace(".zip", ".pkl").replace("ppo_robot", "ppo_robot_vecnormalize")
        if not os.path.exists(vecnorm_path):
            raise FileNotFoundError(f"VecNormalize file not found: {vecnorm_path}\n"
                                    f"Expected alongside checkpoint at: {checkpoint_path}")

        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
        env.norm_obs = True

        model = PPO.load(checkpoint_path, env=env)

    total_rewards = []

    print(f"Obs mean (first 10): {env.obs_rms.mean[:10]}")
    print(f"Obs var  (first 10): {env.obs_rms.var[:10]}")
    print(type(model.policy.action_dist))

    for episode in range(num_episodes):
        obs = env.reset()
        target_pos = np.array(env.get_attr("target_position")[0])
        done = False
        episode_reward = 0.0
        counter = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            reward = float(reward[0])
            done = bool(done[0])

            episode_reward += reward
            counter += 1

            print(f"Step {counter} | Reward: {reward:.4f} | Done: {done}")
            print(f"Action (raw):    {action}")
            print(f"Action (scaled): {(action + 1.0) / 2.0 * TENDON_PARAMS.max_tension}")

        # Fetch rod data once after episode ends
        rod_data = env.get_attr("callback_data_rod_object")[0].copy()
        final_config = rod_data['position'][-1]
        final_pos = np.array([final_config[0][-1], final_config[1][-1], final_config[2][-1]])
        dist = np.linalg.norm(final_pos - target_pos)

        total_rewards.append(episode_reward)
        print(f"\nEpisode {episode + 1}: Reward={episode_reward:.4f} | "
              f"Final dist={dist:.4f} | Target={target_pos}")

        # Creating visualization of episode
        render_episode(
            position_data=rod_data['position'],
            target_position=target_pos,
            output_path=os.path.join(CONFIG.paths.save_dir, "videos", f"test_run_{episode+1}.mp4"),
        )

    avg_reward = sum(total_rewards) / num_episodes
    print(f"\nAverage reward over {num_episodes} episodes: {avg_reward:.4f}")

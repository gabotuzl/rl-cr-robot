import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.cr_env import cr_env
from config import CONFIG


def run_testing(vecnorm_path, checkpoint_path):
    num_episodes = 1
    total_rewards = []

    env_instance = cr_env()
    env = DummyVecEnv([env_instance])
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False
    env.norm_obs = True
    model = PPO.load(checkpoint_path, env=env)


    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        counter = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            target_pos = np.array(env.get_attr("target_position")[0])
            rod_data = env.get_attr("callback_data_rod_object")[0].copy()


            final_config = rod_data['position'][-1]
            final_position_data = np.array((final_config[0][-1], final_config[1][-1], final_config[2][-1]))
            dist = np.linalg.norm(final_position_data-target_pos)


            reward = reward[0]
            done = done[0]
            # print("DIST: ", dist, '\n')
            print("REWARD: ", reward)
            print("DONE: ", done)
            print("COUNTER: ", counter)
            print("ACTION: ", action)
            episode_reward += reward
            counter+=1

            if done:
                target_pos = np.array(env.get_attr("target_position")[0].copy())
                

                break
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Total Reward = {episode_reward}")


        nested_list = [arr.tolist() for arr in rod_data['position']]

        with open(f'{CONFIG.paths.save_dir}/position_data/test_run_{episode+1}.txt', "a") as f:
            f.write(f"{str(nested_list)}")
        with open(f'{CONFIG.paths.save_dir}TARGET_POSITIONS.txt', "a") as f:
            f.write(f"EPISODE {episode+1}, Target pos: {target_pos} \n ")
            # f.write(f"EPISODE {episode+1}, Final pos: {final_position_data} \n\n")
        

    avg_reward = sum(total_rewards) / num_episodes
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
import os
import glob
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import numpy as np


from common.env_utils import make_env
from config import CONFIG
from training.callbacks import make_training_callbacks

def run_training(from_scratch):
    print(f"Creating {CONFIG.ppo.n_envs} environments...")
    env = SubprocVecEnv([make_env(i) for i in range(CONFIG.ppo.n_envs)])

    checkpoint_dir = CONFIG.paths.checkpoint_dir


    if from_scratch is True:

        env = VecNormalize(env, norm_obs=True, norm_reward=True)

        if os.path.exists(checkpoint_dir):
            files = glob.glob(os.path.join(checkpoint_dir, "*"))
            for f in files:
                os.remove(f)
            print("Cleared checkpoint cache.")

        policy_kwargs = dict(
        net_arch=dict(pi=CONFIG.ppo.net_arch_pi, vf=CONFIG.ppo.net_arch_vf),
        ortho_init=True,
        log_std_init=1.0
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
            verbose=1,
            tensorboard_log=CONFIG.paths.tensorboard_dir,
        )

        model.learn(total_timesteps=CONFIG.ppo.total_timesteps, callback=make_training_callbacks())
        model.save(CONFIG.paths.model_save_name)
        env.save(CONFIG.paths.vecnorm_save_name)


    elif from_scratch is False:
        print("Starting from previous checkpoint...")
        model_path = None

        if os.path.exists(checkpoint_dir):
            zips = sorted(glob.glob(os.path.join(checkpoint_dir, "ppo_robot_*_steps.zip")))

            if zips:
                model_path = zips[-1].replace("\\", "/")
                print(f"Resuming from {model_path}...")
                vecnorm_path = model_path.replace(".zip", ".pkl").replace("ppo_robot", "ppo_robot_vecnormalize")
                
                env = VecNormalize.load(vecnorm_path, env)
                print(f"\nObs mean (first 10): {env.obs_rms.mean[:10]}")
                print(f"Obs var  (first 10): {env.obs_rms.var[:10]}")
                assert not np.all(env.obs_rms.var < 1e-6), \
                    "VecNormalize stats look uninitialized — wrong .pkl file?"

                env.training = True
                env.norm_obs = True
                env.norm_reward = True
                print(f"\nVecNormalize flags: training={env.training}, "
                    f"norm_obs={env.norm_obs}, norm_reward={env.norm_reward}")

                model = PPO.load(model_path, env=env, tensorboard_log=CONFIG.paths.tensorboard_dir)
                print(f"\nModel loaded.")
                print(f"  learning_rate: {model.learning_rate}")
                print(f"  n_steps:       {model.n_steps}")
                print(f"  batch_size:    {model.batch_size}")
                print(f"  ent_coef:      {model.ent_coef}")
                print(f"  clip_range:    {model.clip_range}")

        
        steps_done = int(model_path.split("_steps.zip")[0].split("_")[-1])
        remaining = CONFIG.ppo.total_timesteps - steps_done
        print(f"\nSteps done:  {steps_done:>12,}")
        print(f"Steps total: {CONFIG.ppo.total_timesteps:>12,}")
        print(f"Remaining:   {remaining:>12,}")
        assert remaining > 0, "Checkpoint is already at or beyond total_timesteps."
        print(f"\n--- Loading complete ---\n")
        model.learn(reset_num_timesteps=False, total_timesteps=remaining, callback=make_training_callbacks())


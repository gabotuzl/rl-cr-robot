import os
import glob
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize


from common.env_utils import make_env
from config import CONFIG
from training.callbacks import make_training_callbacks

def run_training(from_scratch):
    print(f"Creating {CONFIG.ppo.n_envs} environments...")
    env = SubprocVecEnv([make_env(i) for i in range(CONFIG.ppo.n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    checkpoint_dir = CONFIG.paths.checkpoint_dir


    if from_scratch is True:
        if os.path.exists(checkpoint_dir):
            files = glob.glob(os.path.join(checkpoint_dir, "*"))
            for f in files:
                os.remove(f)
            print("Cleared checkpoint cache.")

        policy_kwargs = dict(
        net_arch=dict(pi=CONFIG.ppo.net_arch_pi, vf=CONFIG.ppo.net_arch_vf),
        ortho_init=True
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


    if from_scratch is False:
        print("Starting from previous checkpoint...")
        model_path = None

        if os.path.exists(checkpoint_dir):
            zips = sorted(glob.glob(os.path.join(checkpoint_dir, "ppo_robot_*_steps.zip")))

            if zips:
                model_path = zips[-1].replace("\\", "/")
                print(f"Resuming from {model_path}...")
                vecnorm_path = model_path.replace(".zip", ".pkl")
                vecnorm_path = vecnorm_path.replace("ppo_robot", "ppo_robot_vecnormalize")
                env = VecNormalize.load(vecnorm_path, env)
                env.training = True
                env.norm_reward = True
                env.norm_obs = True
                model = PPO.load(model_path, env=env)
        
        model.learn(total_timesteps=CONFIG.ppo.total_timesteps, callback=make_training_callbacks)

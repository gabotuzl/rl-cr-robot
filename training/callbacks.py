from stable_baselines3.common.callbacks import (
    CheckpointCallback, BaseCallback, CallbackList
)
from config import CONFIG
import numpy as np


class LoggerCallback(BaseCallback):
    def __init__(self, log_freq: int, verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.update_counter = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            print(
                f"Steps: {self.n_calls:>10,} | "
                f"Total: {self.num_timesteps:>10,} | "
                f"Updates: {self.update_counter}"
            )


        return True

    def _on_rollout_end(self) -> None:
        """Called after each full rollout collection — i.e., each PPO update."""
        self.update_counter += 1
        print(f"\n---- PPO UPDATE #{self.update_counter} ----\n")

class RewardComponentCallback(BaseCallback):
    """
    Logs mean of each reward component per rollout to tensorboard.
    Accumulates values across all env steps during the rollout, then writes means.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.component_buffers = {}
    
    def _on_step(self) -> bool:
        # Called after each env.step() across all parallel envs
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'reward_components' in info:
                for name, value in info['reward_components'].items():
                    if name not in self.component_buffers:
                        self.component_buffers[name] = []
                    self.component_buffers[name].append(value)
        return True
    
    def _on_rollout_end(self) -> None:
        # Called after collecting n_steps * n_envs samples
        for name, values in self.component_buffers.items():
            if values:
                self.logger.record(f'reward_components/{name}_mean', np.mean(values))
                self.logger.record(f'reward_components/{name}_std', np.std(values))
        self.component_buffers = {}  # reset for next rollout


def make_checkpoint_callback(save_path: str = CONFIG.paths.checkpoint_dir, save_freq: int = (CONFIG.ppo.n_steps + 2)) -> CheckpointCallback:
    return CheckpointCallback(
        save_freq=save_freq,
        save_path=save_path,
        name_prefix="ppo_robot",
        save_vecnormalize=True,
    )

def make_training_callbacks(log_freq: int = 5, save_path: str = CONFIG.paths.checkpoint_dir) -> CallbackList:
    return CallbackList([
        make_checkpoint_callback(save_path=save_path),
        LoggerCallback(log_freq=log_freq),
        RewardComponentCallback()
    ])
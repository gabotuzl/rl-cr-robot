from stable_baselines3.common.callbacks import (
    CheckpointCallback, BaseCallback, CallbackList
)

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


def make_checkpoint_callback(save_path: str = "./checkpoints/", save_freq: int = 20_000) -> CheckpointCallback:
    return CheckpointCallback(
        save_freq=save_freq,
        save_path=save_path,
        name_prefix="ppo_robot",
        save_vecnormalize=True,
    )

def make_training_callbacks(log_freq: int = 5, save_path: str = "./checkpoints/") -> CallbackList:
    return CallbackList([
        make_checkpoint_callback(save_path=save_path),
        LoggerCallback(log_freq=log_freq),
    ])
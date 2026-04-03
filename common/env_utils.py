from env.cr_env import cr_env
from stable_baselines3.common.monitor import Monitor
from config import CONFIG


def make_env(rank=0):
    def _init():
        env = cr_env()  
        env = Monitor(env, filename=f"{CONFIG.paths.log_dir}env_{rank}")
        return env
    return _init
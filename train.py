"""
    This script wraps the default rllib train script and registers the CavalryVsInfantry scenario.
"""

from ray.rllib.train import create_parser, run
from ray.tune.registry import register_env
import ray
from ray import tune
#from cav_vs_inf_env import *
from cur_cav_vs_inf_env import *

register_env('CavalryVsInfantry', lambda c: CavalryVsInfantryEnv(c))
register_env('SimpleMinimapCavVsInf', lambda c: SimpleMinimapCavVsInfEnv(c))
register_env('MinimapCavVsInf', lambda c: MinimapCavVsInfEnv(c))

# Curriculum learning goals
REWARD_MEAN_GOAL = -10
level = 1

def on_train_result(info):
    global level
    result = info["result"]
    if result["episode_reward_mean"] > REWARD_MEAN_GOAL:
        print("increasing level!")
        level += 1
    trainer = info["trainer"]
    trainer.workers.foreach_worker(
        lambda ev: ev.foreach_env(
            lambda env: env.set_cur_level(level)))

if __name__ == '__main__':
    ray.init(num_gpus=0)
    tune.run(
        "PPO",
        name="tyler_PPO_minimapcavvsinf_curric",
        checkpoint_freq=100,
        checkpoint_at_end=True,
        config={
            "env": "MinimapCavVsInf",
            "num_gpus": 0,
            "num_workers": 3,
            "eager": False,
            "callbacks": {
                    "on_train_result": on_train_result,
                },
        },
    )

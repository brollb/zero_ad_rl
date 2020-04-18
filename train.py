"""
    This script wraps the default rllib train script and registers the CavalryVsInfantry scenario.
"""

from ray.rllib.train import create_parser, run
from ray.tune.registry import register_env
import ray
from ray import tune
from cav_vs_inf_env import *

register_env('CavalryVsInfantry', lambda c: CavalryVsInfantryEnv(c))
register_env('SimpleMinimapCavVsInf', lambda c: SimpleMinimapCavVsInfEnv(c))
register_env('MinimapCavVsInf', lambda c: MinimapCavVsInfEnv(c))

if __name__ == '__main__':
    ray.init(num_gpus=1)
    tune.run(
        "DQN",
        name="tyler_DQN_minimapcavvsinf0908",
        checkpoint_freq=100,
        stop={"episode_reward_mean": -20},
        config={
            "env": "MinimapCavVsInf",
            "num_gpus": 1,
            "num_workers": 3,
            "eager": False,
        },
        #num_samples=10,
        #resources_per_trial={"cpu": 4, "gpu":0.25},
    )
    #parser = create_parser()
    #parser.set_defaults(env='CavalryVsInfantry')
    #args = parser.parse_args()
    #print(args)
    #run(args, parser)

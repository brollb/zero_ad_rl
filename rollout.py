"""
    This script wraps the rllib rollout command but uses the custom environment.
"""
from ray.rllib.rollout import create_parser, run
from ray.tune.registry import register_env
from cav_vs_inf_env import *

register_env('CavalryVsInfantry', lambda c: CavalryVsInfantryEnv(c))
register_env('SimpleMinimapCavVsInf', lambda c: SimpleMinimapCavVsInfEnv(c))
register_env('MinimapCavVsInf', lambda c: MinimapCavVsInfEnv(c))

if __name__ == '__main__':
    parser = create_parser()
    parser.set_defaults(env='CavalryVsInfantry', no_render=True)
    args = parser.parse_args()
    run(args, parser)

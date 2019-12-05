"""
    This script wraps the rllib rollout command but uses the custom environment.
"""
from ray.rllib.rollout import create_parser, run
from ray.tune.registry import register_env
from cav_vs_inf_env import CavalryVsInfantryEnv

register_env('CavalryVsInfantry', lambda c: CavalryVsInfantryEnv(c))

if __name__ == '__main__':
    parser = create_parser()
    parser.set_defaults(env='CavalryVsInfantry')
    args = parser.parse_args()
    run(args, parser)

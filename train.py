"""
    This script wraps the default rllib train script and registers the CavalryVsInfantry scenario.
"""

from ray.rllib.train import create_parser, run
from ray.tune.registry import register_env
from cav_vs_inf_env import CavalryVsInfantryEnv

register_env('CavalryVsInfantry', lambda c: CavalryVsInfantryEnv(c))

if __name__ == '__main__':
    parser = create_parser()
    parser.set_defaults(env='CavalryVsInfantry')
    args = parser.parse_args()
    run(args, parser)

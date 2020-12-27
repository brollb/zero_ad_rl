"""
    This script wraps the rllib rollout command but uses the custom environment.
"""
from ray.rllib.rollout import create_parser, run
from ray.tune.registry import register_env
from .env import register_envs

register_envs()

if __name__ == '__main__':
    parser = create_parser()
    parser.set_defaults(env='CavalryVsInfantry', no_render=True)
    args = parser.parse_args()
    run(args, parser)

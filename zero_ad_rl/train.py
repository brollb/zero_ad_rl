"""
    This script wraps the default rllib train script and registers the CavalryVsInfantry scenario.
"""

from ray.rllib import train
from ray.tune.registry import register_env
from .env import register_envs

register_envs()

def invoke_if_defined(obj, method, arg):
    fn = getattr(obj, method, None)
    if fn is not None:
        fn(arg)


def on_train_result(info):
    result = info["result"]
    reward_mean = result["episode_reward_mean"]
    trainer = info["trainer"]
    trainer.workers.foreach_worker(
        lambda ev: ev.foreach_env(
            lambda env: invoke_if_defined(env, 'on_train_result', reward_mean)))

def run(args, parser):
    config = args.config

    if 'callbacks' not in config:
        config['callbacks'] = {}

    if 'on_train_result' not in config['callbacks']:
         config['callbacks']['on_train_result'] = on_train_result
    else:
        print('on_train_result defined. Overriding default')
    train.run(args, parser)

def create_parser():
    parser = train.create_parser()
    parser.set_defaults(env='CavalryVsInfantry')
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)

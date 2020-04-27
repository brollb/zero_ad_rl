"""
    This script wraps the default rllib train script and registers the CavalryVsInfantry scenario.
"""

from ray.rllib.train import create_parser, run
from ray.tune.registry import register_env
from cav_vs_inf_env import *

register_env('CavalryVsInfantry', lambda c: CavalryVsInfantryEnv(c))
register_env('SimpleMinimapCavVsInf', lambda c: SimpleMinimapCavVsInfEnv(c))
register_env('MinimapCavVsInf', lambda c: MinimapCavVsInfEnv(c))

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

if __name__ == '__main__':
    parser = create_parser()
    parser.set_defaults(env='CavalryVsInfantry')
    args = parser.parse_args()
    config = args.config

    if 'callbacks' not in config:
        config['callbacks'] = {}

    if 'on_train_result' not in config['callbacks']:
         config['callbacks']['on_train_result'] = on_train_result
    else:
        print('on_train_result defined. Overriding default')
    run(args, parser)

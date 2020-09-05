from zero_ad import GameState
import ray
import gym
from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env, _global_registry, ENV_CREATOR
from cav_vs_inf_env import *
import argparse
import json
import pickle

register_env('CavalryVsInfantry', lambda c: CavalryVsInfantryEnv(c))
register_env('SimpleMinimapCavVsInf', lambda c: SimpleMinimapCavVsInfEnv(c))
register_env('MinimapCavVsInf', lambda c: MinimapCavVsInfEnv(c))

def walk_target(command):
    return (command['x'], command['z'])

def distance(p1, p2):
    return math.sqrt(sum((math.pow(x2 - x1, 2) for (x1, x2) in zip(p1, p2))))

def closest_action(env, command):
    if command['type'] == 'attack':
        return 8

    possible_actions = [(i, env.resolve_action(i)) for i in range(8)]
    possible_actions.sort(key=lambda pair: distance(walk_target(pair[1]), walk_target(command)))
    return possible_actions[0][0]

def is_game_over(state):
    return any([player['state'] != 'active' for player in state.data['players']])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('states', help='states.jsonl file to generate demonstration from')  # TODO: var args?
    parser.add_argument('--run')
    parser.add_argument('--env')
    parser.add_argument('--url', default='http://127.0.0.1:6000',
            help='0 AD game server URL (running with --rlinterface flag)')
    parser.add_argument('--out', default='demonstrations')
    parser.add_argument('--target-env')

    args = parser.parse_args()

    ray.init()
    cls = get_agent_class(args.run)
    agent = cls(env=args.env)
    agent.restore(args.checkpoint)

    env = agent.workers.local_worker().env
    if args.target_env:
        env_creator = _global_registry.get(ENV_CREATOR, args.target_env)
        target_env = env_creator(env.config)
    else:
        target_env = None

    trajectories = []
    prev_obs = None
    prev_action = None
    with open(args.states, 'r') as states_file:
        states = (GameState(json.loads(line), env.game) for line in states_file)
        trajectory = []
        for state in states:
            if is_game_over(state):
                continue

            env.prev_state = env.state
            env.state = state
            obs = env.observation(state)
            action = agent.compute_action(obs)
            command = env.resolve_action(action)
            if target_env:
                target_env.prev_state = target_env.state
                target_env.state = state
                obs = target_env.observation(state)
                action = closest_action(target_env, command)

            if prev_obs is not None:
                trajectory.append([prev_obs, prev_action, obs])

            prev_obs = obs
            prev_action = action
        trajectories.append(trajectory)

    pickle.dump(trajectories, open(args.out, 'wb'))

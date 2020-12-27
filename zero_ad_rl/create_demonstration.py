from zero_ad import GameState
import ray
import gym
from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env, _global_registry, ENV_CREATOR
from .env import register_envs
import argparse
import json
import pickle
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter

register_envs()

def walk_target(command):
    return (command['x'], command['z'])

def distance(p1, p2):
    return math.sqrt(sum((math.pow(x2 - x1, 2) for (x1, x2) in zip(p1, p2))))

def closest_action(env, command):
    if command['type'] == 'attack':
        return 8

    # FIXME: These should be relative first!!
    possible_actions = [(i, env.actions.to_json(i, env.state)) for i in range(8)]
    possible_actions.sort(key=lambda pair: distance(walk_target(pair[1]), walk_target(command)))
    return possible_actions[0][0]

def is_game_over(state):
    return any([player['state'] != 'active' for player in state.data['players']])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('states', help='states.jsonl file to generate demonstration from', nargs='*')
    parser.add_argument('--run')
    parser.add_argument('--env')
    parser.add_argument('--url', default='http://127.0.0.1:6000',
            help='0 AD game server URL (running with --rlinterface flag)')
    parser.add_argument('--out', default='demonstrations')
    parser.add_argument('--target-env')
    parser.add_argument('--skip-count', default=8, type=int)

    args = parser.parse_args()

    ray.init()
    cls = get_agent_class(args.run)
    agent = cls(env=args.env)
    agent.restore(args.checkpoint)

    # TODO: Refactor this...
    env = agent.workers.local_worker().env
    if args.target_env:
        env_creator = _global_registry.get(ENV_CREATOR, args.target_env)
        target_env = env_creator(env.config)
    else:
        target_env = None

    #create_demonstration(agent, states_path, target_env)  # FIXME: finish this

    prev_obs = None
    prev_action = None
    batch_builder = SampleBatchBuilder()
    writer = JsonWriter(args.out)
    debug_count = 0
    for states_path in args.states:
        with open(states_path, 'r') as states_file:
            states_file.readline()  # skip the first
            lines = (line for (index, line) in enumerate(states_file) if index % args.skip_count == 0)
            states = (GameState(json.loads(line), env.game) for line in lines)
            trajectory = []
            for (t, state) in enumerate(states):
                if is_game_over(state):
                    continue

                env.prev_state = env.state
                env.state = state
                obs = env.observation(state)
                action = agent.compute_action(obs)
                #print(obs, action)
                command = env.actions.to_json(action, state)
                if target_env:
                    target_env.prev_state = target_env.state
                    target_env.state = state
                    obs = target_env.observation(state)
                    action = closest_action(target_env, command)

                if prev_obs is not None:
                    trajectory.append([prev_obs, prev_action, obs])
                    # FIXME: Am I using this incorrectly?
                    print('prev_obs', prev_obs)
                    print('action', action)
                    print('prev_action', prev_action)
                    batch_builder.add_values(
                        t=t,
                        eps_id=0,
                        agent_index=0,
                        obs=prev_obs,
                        actions=action,
                        action_prob=1.0,  # TODO: put the true action probability here
                        rewards=0,
                        prev_actions=prev_action,
                        prev_rewards=0,
                        dones=False,  # TODO
                        infos=None,
                        new_obs=obs)

                prev_obs = obs
                prev_action = action
                debug_count += 0
        writer.write(batch_builder.build_and_reset())

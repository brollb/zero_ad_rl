import ray
import argparse
from ray.rllib.offline.json_reader import JsonReader
from .env import register_envs
from .agent import get_agent_class

def count_lines(filename):
    line_count = 0
    with open(filename, 'r') as f:
        for _ in f:
            line_count += 1
    return line_count


def count_batches(files):
    return sum((count_lines(filename) for filename in files))

def score_demonstrations(agent, files, verbose=False):
    reader = JsonReader(files)
    num_batches = count_batches(files)
    correct = 0
    total = 0
    for _ in range(num_batches):
        batch = reader.next()
        for (i, obs) in enumerate(batch['obs']):
            action = agent.compute_action(obs)
            expected_action = batch['actions'][i]
            if action == expected_action:
                correct += 1
            total += 1
            if verbose:
                print(f'{obs} {action}')

    return correct, total

if __name__ == '__main__':
    register_envs()

    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('files', help='demonstrations to use for scoring', nargs='+')
    parser.add_argument('--run')
    parser.add_argument('--env')
    parser.add_argument('--url', default='http://127.0.0.1:6000',
            help='0 AD game server URL (running with --rlinterface flag)')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    ray.init()
    cls = get_agent_class(args.run)
    agent = cls(env=args.env)
    agent.restore(args.checkpoint)

    env = agent.workers.local_worker().env

    correct, total = score_demonstrations(agent, args.files, args.verbose)
    print(f'{correct/total} ({correct}/{total})')

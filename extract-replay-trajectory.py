import zero_ad
import argparse
import os
from os import path
import json

def extract_config(commands_file):
    pos = commands_file.tell()
    commands_file.seek(0)
    config = commands_file.readline()[6:]
    commands_file.seek(pos)
    return config

def extract_actions(commands_file):
    turn = -1
    cmds = []
    players = []
    for line in commands_file.readlines():
        if line.startswith('start'):
            config = line[6:]
            #state = game.reset(config)  # TODO: should I skip the next turn?
            #print(json.dumps(state.data))
        elif line.startswith('turn'):
            pass
        elif line.startswith('cmd'):
            player = line[4]
            cmd = line[6:]
            players.append(player)
            cmds.append(json.loads(cmd))
        elif line.startswith('end'):
            yield (cmds, players)
            cmds = []
            players = []
        else:
            raise Exception(f'Unrecognized command: {line}')

def extract_trajectory(commands_file):
    config = extract_config(commands_file)
    turn = -1
    cmds = []
    players = []
    yield (game.reset(config), [], [])  # TODO: should I skip the next turn?
    for (cmds, players) in extract_actions(commands_file):
        yield (game.step(cmds, players), cmds, players)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('replay_dir')
    parser.add_argument('--url', default='http://127.0.0.1:6000',
            help='0 AD game server URL (running with --rlinterface flag)')
    parser.add_argument('--outdir', help='output directory')
    parser.add_argument('--actions', default='actions.jsonl', help='actions file name')
    parser.add_argument('--player', default='1', help='log actions of the given player')

    args = parser.parse_args()
    game = zero_ad.ZeroAD(args.url)

    commands_path = path.join(args.replay_dir, 'commands.txt')
    commands_file = open(commands_path, 'r')

    outdir = args.outdir or path.basename(args.replay_dir)
    os.makedirs(outdir, exist_ok=True)
    states_file = open(path.join(outdir, 'states.jsonl'), 'w')
    actions_file = open(path.join(outdir, args.actions or 'actions.jsonl'), 'w')
    for (state, cmds, players) in extract_trajectory(commands_file):
        states_file.write(json.dumps(state.data) + '\n')
        actions = [ cmd for (cmd, player_id) in zip(cmds, players) if player_id == args.player ]
        actions_file.write(json.dumps(actions) + '\n')

    states_file.close()
    actions_file.close()
    commands_file.close()

from .env import AttackRetreat, AttackAndMove
import numpy as np
import zero_ad
import argparse
import math
import json
import pickle
import os
from os import path
from PIL import ImageDraw, Image

def walk_target(command):
    return (command['x'], command['z'])

def distance(p1, p2):
    return math.sqrt(sum((math.pow(x2 - x1, 2) for (x1, x2) in zip(p1, p2))))

def closest_action(actions, state, command):
    if command['type'] == 'attack':
        return 8

    # FIXME: These should be relative first!!
    possible_actions = [(i, actions.to_json(i, state)) for i in range(8)]
    possible_actions.sort(key=lambda pair: distance(walk_target(pair[1]), walk_target(command)))

    return possible_actions[0][0]

def center(units):
    positions = np.array([unit.position() for unit in units])
    return np.mean(positions, axis=0)

def render_move_action(state, command, color='#777777', image=None):
    if command['type'] != 'walk':
        raise Exception(f'Unexpected command type: {command["type"]}')

    if image is None:
        image = Image.fromarray(np.zeros((84, 84, 3)).astype(np.uint8))

    center_pt = center(state.units(owner=1))
    target = np.array([command['x'], command['z']])
    rel_target = target-center_pt + 42
    draw = ImageDraw.Draw(image)
    # TODO: for each action, render stuff
    draw.line((42, 42, rel_target[0], rel_target[1]), fill=color)
    return image
    # TODO: draw an arrow in the given direction? Or a gradient line?

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('states', help='states.jsonl file to generate demonstration from')
    parser.add_argument('--url', default='http://127.0.0.1:6000',
            help='0 AD game server URL (running with --rlinterface flag)')
    parser.add_argument('--out', default=None)

    args = parser.parse_args()

    outdir = args.out
    if args.out is None:
        outdir = 'viz-actions-' + path.dirname(args.states)

    os.makedirs(outdir, exist_ok=True)
    game = zero_ad.ZeroAD(args.url)
    simple_builder = AttackRetreat()
    builder = AttackAndMove()
    with open(args.states, 'r') as states_file:
        jsons = ( json.loads(line) for line in states_file if line.strip() )
        states = ( zero_ad.GameState(json_data, game) for json_data in jsons )
        states_actions = ( (simple_builder.to_json(0, state), state) for state in states )
        for (i, (retreat, state)) in enumerate(states_actions):
            possible_actions = [builder.to_json(i, state) for i in range(8)]
            image = render_move_action(state, retreat, '#0000ff')
            for action_json in possible_actions:
                image = render_move_action(state, action_json, image=image)

            new_action = closest_action(builder, state, retreat)
            new_action_json = builder.to_json(new_action, state)

            image = render_move_action(state, new_action_json, '#ff0000', image)
            image.save(path.join(outdir, f'action-{i}.png'), format='PNG')

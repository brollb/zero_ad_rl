# TODO: For now, assume minimap states
from cav_vs_inf_env import Minimap
import zero_ad
import argparse
import json
import pickle
import os
from os import path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('states', help='states.jsonl file to generate demonstration from')
    parser.add_argument('--url', default='http://127.0.0.1:6000',
            help='0 AD game server URL (running with --rlinterface flag)')
    parser.add_argument('--out', default=None)

    args = parser.parse_args()

    outdir = args.out
    if args.out is None:
        outdir = 'viz-' + path.dirname(args.states)

    os.makedirs(outdir, exist_ok=True)
    game = zero_ad.ZeroAD(args.url)
    builder = Minimap()
    with open(args.states, 'r') as states_file:
        jsons = ( json.loads(line) for line in states_file if line.strip() )
        states = ( zero_ad.GameState(json_data, game) for json_data in jsons )
        images = ( builder.to_image(state) for state in states )
        for (i, image) in enumerate(images):
            image.save(path.join(outdir, f'state-{i}.png'), format='PNG')

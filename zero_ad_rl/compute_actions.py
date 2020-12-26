import argparse

# TODO: [ ] Get the model
# TODO: [ ] Call the forward pass
# TODO: [ ] Save the game actions as 0 AD commands
# TODO: [ ] "Generalizing Behavior Cloning to Heterogeneous Action Spaces"?

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TODO: need to know the state, action representation (ie, the env)
    parser.add_argument('model', help='model to use to compute actions')
    parser.add_argument('states', help='states.jsonl')
    parser.add_argument('--outfile', help='output file for actions')

    args = parser.parse_args()

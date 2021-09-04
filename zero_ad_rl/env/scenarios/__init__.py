from os import path

def load_config(name):
    filename = name + '.json'
    configs_dir = path.dirname(path.realpath(__file__))
    config_path = path.join(configs_dir, filename)
    with open(config_path) as f:
        config = f.read()
    return config

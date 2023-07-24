''' Configuration for experiments.
'''
import json
from argparse import Namespace


def get_config(config_filepath):
    with open(config_filepath, 'r') as config_file:
        conf = json.load(config_file, object_hook=lambda d: Namespace(**d))
    return conf
if __name__ == "__main__":
    conf=get_config('./config.json')
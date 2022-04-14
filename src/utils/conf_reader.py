import argparse
from pathlib import Path

import torch
import yaml


def _read_conf_file(name):
    """ Read the config.yaml file """
    try:
        file = yaml.load(open(Path('configs') / name), Loader=yaml.FullLoader)
        return file
    except FileNotFoundError:
        raise FileNotFoundError("Config file not found")


def get_config():
    """ Parse the arguments and merge them with the config file """

    conf = _read_conf_file('base_config.yaml')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    conf['device'] = device

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=None, help="The learning rate")
    parser.add_argument("--load_weights", default=None, help="name of the model to load")
    parser.add_argument("--batch_size", default=32, help="The mini-batch size")
    args = parser.parse_args()

    if args.lr is not None:
        conf['optimizer']['lr'] = float(args.lr)
    if args.batch_size is not None:
        conf['train']['batch_size'] = float(args.batch_size)
    conf['load_weights'] = str(args.load_weights)

    conf['train']['batch_size'] = int(args.batch_size)
    conf['env']['use_data_parallel'] = 'cuda' in device and conf['env']['world_size'] > 1

    return conf

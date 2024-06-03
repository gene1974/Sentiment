import argparse
import configparser

def parse_args():
    args = {}

    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read('./config.ini')
    for key in config['DEFAULT']:
        args[key] = config['DEFAULT'][key]
    # args = dict(config['DEFAULT'])
    print(args)

    return args

if __name__ == '__main__':
    parse_args()

import argparse
import configparser
import logging
from humanfriendly import parse_size, parse_timespan
from numpy import random
from storage import Backup, Node


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="client_server.cfg", help="configuration file")
    parser.add_argument("--max-t", default="100 years")
    parser.add_argument("--seed", default=1, help="random seed")
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)  # set a seed to make experiments repeatable
    if args.verbose:
        logging.basicConfig(format='{levelname}:{message}', level=logging.INFO, style='{')  # output info on stdout

    # functions to parse every parameter of peer configuration
    parsing_functions = [
        ('n', int), ('k', int),
        ('data_size', parse_size), ('storage_size', parse_size),
        ('upload_speed', parse_size), ('download_speed', parse_size),
        ('average_uptime', parse_timespan), ('average_downtime', parse_timespan),
        ('average_lifetime', parse_timespan), ('average_recover_time', parse_timespan),
        ('arrival_time', parse_timespan)
    ]
    config = configparser.ConfigParser()
    config.read(args.config)
    nodes = []  # we build the list of nodes to pass to the Backup class
    for node_class in config.sections():
        class_config = config[node_class]
        # list comprehension: https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions
        cfg = [parse(class_config[name]) for name, parse in parsing_functions]
        # the `callable(p1, p2, *args)` idiom is equivalent to `callable(p1, p2, args[0], args[1], ...)
        nodes.extend(Node(f"{node_class}-{i}", *cfg) for i in range(class_config.getint('number')))
    # uncomment to make the first client selfish, note that the simulation will
    # stop after some time as the servers will have no space to store the blocks
    # nodes[0].selfish = True
    sim = Backup(nodes)
    sim.run(parse_timespan(args.max_t))
    for node in nodes:
        if "client" in node.name:
            print(f"{node.name}: local blocks: {len(node.local_blocks)},"
                  f" backed up blocks: {len(node.backed_up_blocks)}, remote blocks: {len(node.remote_blocks_held)}")
        else:
            print(f"{node.name}: remote blocks: {len(node.remote_blocks_held)}")

    sim.log_info(f"Simulation over")

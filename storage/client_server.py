import argparse
import configparser
import logging
from humanfriendly import parse_size, parse_timespan
from matplotlib import pyplot as plt
from numpy import random
from storage import Backup, Node, get_lost_blocks_count


def run():
    simulation_count = 1
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
    life_time_list = ["1 year", "16 years", "81 years", "256 years", "625 years",
                      "1296 years"]  # n^4 years: 1 <= n <= 6
    lost_blocks_arr: list[int] = []
    lost_blocks_avg_list = []
    data_for_lost_blocks_avg_plot = []
    for life_time in life_time_list:
        for i in range(simulation_count):
            nodes = []  # we build the list of nodes to pass to the Backup class
            config.set("DEFAULT", "average_lifetime", life_time)
            for node_class in config.sections():
                class_config = config[node_class]
                # list comprehension: https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions
                cfg = [parse(class_config[name]) for name, parse in parsing_functions]
                # the `callable(p1, p2, *args)` idiom is equivalent to `callable(p1, p2, args[0], args[1], ...)
                nodes.extend(Node(f"{node_class}-{i}", *cfg) for i in range(class_config.getint('number')))
            sim = Backup(nodes)
            sim.run(parse_timespan(args.max_t))
            lost_blocks_arr.append(get_lost_blocks_count(sim.nodes))
            sim.log_info(f"Simulation over")
        lost_blocks_avg_list.append(sum(lost_blocks_arr) / len(lost_blocks_arr))
    data_for_lost_blocks_avg_plot.append(lost_blocks_avg_list)
    for i in data_for_lost_blocks_avg_plot:
        plt.plot(life_time_list, i,
                 label=f"Simulation")
    plt.xlabel("Average lifetime of nodes")
    plt.ylabel("Average lost blocks")
    plt.title(f"Results of {simulation_count} times simulation: Client-Server")
    plt.legend(loc=0)
    plt.grid()
    plt.show()

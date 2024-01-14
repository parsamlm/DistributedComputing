#!/usr/bin/env python3

import argparse
import csv
import collections
import logging
from random import expovariate, seed

from discrete_event_sim import Simulation, Event

# One possible modification is to use a different distribution for job sizes or and/or interarrival times.
# Weibull distributions (https://en.wikipedia.org/wiki/Weibull_distribution) are a generalization of the
# exponential distribution, and can be used to see what happens when values are more uniform (shape > 1,
# approaching a "bell curve") or less (shape < 1, "heavy tailed" case when most of the work is concentrated
# on few jobs).

# To use weibull variates, for a given set of parameters do something like
# from workloads import weibull_generator
# gen = weibull_generator(shape, mean)

# and then call gen() every time you need a random variable


class MMN(Simulation):

    def __init__(self, lambd, mu, n, d):
        if n != 1 or d != 1:
            raise NotImplementedError  # extend this to make it work for multiple queues and supermarket

        super().__init__()
        self.running = None  # if not None, the id of the running job
        self.queue = collections.deque()  # FIFO queue of the system
        self.arrivals = {}  # dictionary mapping job id to arrival time
        self.completions = {}  # dictionary mapping job id to completion time
        self.lambd = lambd
        self.n = n
        self.mu = mu
        self.arrival_rate = lambd * n
        self.schedule(expovariate(self.arrival_rate), Arrival(0))

    def schedule_arrival(self, job_id):  # TODO: complete this method
        # schedule the arrival following an exponential distribution, to compensate the number of queues the arrival
        # time should depend also on "n"
        self.schedule(..., ...)

    def schedule_completion(self, job_id):  # TODO: complete this method
        # schedule the time of the completion event
        self.schedule(..., ...)

    @property
    def queue_len(self):
        return (self.running is not None) + len(self.queue)


class Arrival(Event):

    def __init__(self, job_id):
        self.id = job_id

    def process(self, sim: MMN):  # TODO: complete this method
        # set the arrival time of the job
        ...
        # if there is no running job, assign the incoming one and schedule its completion
        if ...:
            ...
            ...
        # otherwise put the job into the queue
        else:
            ...
        # schedule the arrival of the next job
        ...

class Completion(Event):
    def __init__(self, job_id):
        self.id = job_id  # currently unused, might be useful when extending

    def process(self, sim: MMN):  # TODO: complete this method
        assert sim.running is not None
        # set the completion time of the running job
        ...
        # if the queue is not empty
        if ...:
            # get a job from the queue
            ...
            # schedule its completion
            ...
        else:
            sim.running = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambd', type=float, default=0.7)
    parser.add_argument('--mu', type=float, default=1)
    parser.add_argument('--max-t', type=float, default=1_000_000)
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--d', type=int, default=1)
    parser.add_argument('--csv', help="CSV file in which to store results")
    parser.add_argument("--seed", help="random seed")
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)  # set a seed to make experiments repeatable
    if args.verbose:
        logging.basicConfig(format='{levelname}:{message}', level=logging.INFO, style='{')  # output info on stdout

    sim = MMN(args.lambd, args.mu, args.n, args.d)
    sim.run(args.max_t)

    completions = sim.completions
    W = (sum(completions.values()) - sum(sim.arrivals[job_id] for job_id in completions)) / len(completions)
    print(f"Average time spent in the system: {W}")
    print(f"Theoretical expectation for random server choice: {1 / (1 - args.lambd)}")

    if args.csv is not None:
        with open(args.csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([args.lambd, args.mu, args.max_t, W])


if __name__ == '__main__':
    main()

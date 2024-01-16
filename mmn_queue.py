#!/usr/bin/env python3

import argparse
import csv
import collections
import logging
from random import expovariate, seed
import matplotlib.pyplot as plt

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
        if d != 1:
            raise NotImplementedError  # extend this to make it work for multiple queues and supermarket

        super().__init__()
        self.running = []  # create an array of servers
        self.queue = collections.deque()  # FIFO queue of the system
        self.arrivals = {}  # dictionary mapping job id to arrival time
        self.completions = {}  # dictionary mapping job id to completion time
        self.lambd = lambd
        self.n = n
        self.mu = mu
        self.arrival_rate = lambd * n
        self.schedule(expovariate(self.arrival_rate), Arrival(0))
        self.timeInterval = 1000
        self.schedule(self.timeInterval, QueLength())
        self.queueLengths = []

    def schedule_arrival(self, job_id):  # TODO: complete this method
        # schedule the arrival following an exponential distribution, to compensate the number of queues the arrival
        # time should depend also on "n"
        self.schedule(expovariate(self.arrival_rate), Arrival(job_id))

    def schedule_completion(self, job_id):  # TODO: complete this method
        # schedule the time of the completion event
        self.schedule(expovariate(self.mu), Completion(job_id))

    @property
    def queue_len(self):
        return len(self.running) + len(self.queue)

class QueLength(Event):
    def process(self, sim: MMN):
        sim.queueLengths.append(sim.queue_len)
        sim.schedule(sim.timeInterval, QueLength())

class Arrival(Event):

    def __init__(self, job_id):
        self.id = job_id

    def process(self, sim: MMN):  # TODO: complete this method
        # print(f"Job {self.id} arrived at time {sim.t}")
        # set the arrival time of the job
        sim.arrivals[self.id] = sim.t
        # if there is no running job, assign the incoming one and schedule its completion
        if len(sim.running) < sim.n:
            sim.running.append(self.id)
            sim.schedule_completion(self.id)
        # otherwise put the job into the queue
        else:
            sim.queue.append(self.id)
        # schedule the arrival of the next job
        sim.schedule_arrival(self.id+1)

class Completion(Event):
    def __init__(self, job_id):
        self.id = job_id  # currently unused, might be useful when extending

    def process(self, sim: MMN):  # TODO: complete this method
        # print(f"Job {self.id} completed at time {sim.t}")
        server_index = sim.running.index(self.id)
        assert server_index is not None
        # set the completion time of the running job
        sim.completions[self.id] = sim.t
        # if the queue is not empty
        if len(sim.queue) > 0:
            # get a job from the queue
            job = sim.queue.popleft()
            sim.running[server_index] = job
            # schedule its completion
            sim.schedule_completion(job)
        else:
            sim.running.remove(self.id)

def run_simulation(lambd, mu, n, max_t, d):
    sim = MMN(lambd, mu, n, d)
    sim.run(max_t)
    completions = sim.completions
    W = (sum(completions.values()) - sum(sim.arrivals[job_id] for job_id in completions)) / len(completions)
    print(f"Average time spent in the system: {W}")

    return sim.queueLengths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambd', type=float, default=0.5)
    parser.add_argument('--mu', type=float, default=1)
    parser.add_argument('--max-t', type=float, default=1_000_000)
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--d', type=int, default=1)
    parser.add_argument('--csv', help="CSV file in which to store results")
    parser.add_argument("--seed", help="random seed")
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()

    if args.seed:
        seed(args.seed)  # set a seed to make experiments repeatable
    if args.verbose:
        logging.basicConfig(format='{levelname}:{message}', level=logging.INFO, style='{')  # output info on stdout

    sim = MMN(args.lambd, args.mu, args.n, args.d)
    sim.run(args.max_t)

    completions = sim.completions
    W = (sum(completions.values()) - sum(sim.arrivals[job_id] for job_id in completions)) / len(completions)
    print(f"Average time spent in the system: {W}")
    # print(f"Theoretical expectation for waiting time: {args.lambd / (1 - args.lambd)}")
    print(f"Theoretical expectation for random server choice: {1 / (1 - args.lambd)}")

    for lambd in [0.5, 0.9, 0.95, 0.99]:
        queueLengths = run_simulation(lambd, args.mu, args.n, args.max_t, args.d)
        counts = [0]*15
        for length in queueLengths:
            if length == 0:  # Skip over queue lengths of zero
                continue
            for i in range(min(length+1, 15)):
                counts[i] += 1
        fractions = [count/len(queueLengths) for count in counts]
        plt.plot(range(1, 15), fractions[1:], label=f'lambd={lambd}')

    plt.xlabel('Queue length')
    plt.ylabel('Fraction of queues with at least that size')
    plt.xlim(0, 15)  # Set x-axis limits to 0 and 15
    plt.ylim(0.0, 1.0)  # Set y-axis limits to 0.0 and 1.0
    plt.legend()
    plt.grid(True)
    plt.show()
    
    if args.csv is not None:
        with open(args.csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([args.lambd, args.mu, args.max_t, W])


if __name__ == '__main__':
    main()

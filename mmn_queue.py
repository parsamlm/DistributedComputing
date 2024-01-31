#!/usr/bin/env python3

import argparse
import csv
import collections
import logging
from random import expovariate, seed, sample
from workloads import weibull_generator
import matplotlib.pyplot as plt

from discrete_event_sim import Simulation, Event
from multiprocessing import Pool

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

    def __init__(self, lambd, mu, n, d, timeSlice = 1):
        super().__init__()
        self.running = [None] * n  # create an array of servers
        self.queues = [collections.deque() for _ in range(n)]  # FIFO queue of the system
        self.arrivals = {}  # dictionary mapping job id to arrival time
        self.completions = {}  # dictionary mapping job id to completion time
        self.lambd = lambd
        self.n = n
        self.mu = mu
        self.d = d
        self.arrival_rate = lambd * n
        self.schedule(0, Arrival(0))
        self.timeInterval = 1000
        self.schedule(self.timeInterval, QueLength())
        self.queueLengths = []  # Create a list of lists to store the length of each queue
        self.timeSlice = timeSlice

    def schedule_arrival(self, job_id):  # TODO: complete this method
        # schedule the arrival following an exponential distribution, to compensate the number of queues the arrival
        # time should depend also on "n"
        self.schedule(expovariate(self.arrival_rate), Arrival(job_id))

    def schedule_completion(self, job_id, server, remaining_time = None):  # TODO: complete this method
        service_time = expovariate(self.mu)
        # schedule the time of the completion event
        if remaining_time is not None:
            service_time = remaining_time

        if service_time > self.timeSlice:
            self.schedule(self.timeSlice, Preemption(job_id, server, service_time - self.timeSlice))
        else:        
            self.schedule(service_time, Completion(job_id, server))

    def queue_len(self, i):
        return len(self.queues[i]) + (1 if self.running[i] is not None else 0)

class QueLength(Event):
    def process(self, sim: MMN):
        for i in range(sim.n):
            sim.queueLengths.append(sim.queue_len(i))
        sim.schedule(sim.timeInterval, QueLength())

class Arrival(Event):

    def __init__(self, job_id):
        self.id = job_id

    def process(self, sim: MMN):  # TODO: complete this method
        # print(f"Job {self.id} arrived at time {sim.t}")
        # set the arrival time of the job
        sim.arrivals[self.id] = sim.t
        servers = sample(range(sim.n), sim.d)  # sample d servers
        server = min(servers, key=lambda s: len(sim.queues[s])) # find the server with the shortest queue
        # if there is no running job, assign the incoming one and schedule its completion
        if sim.running[server] is None:
            sim.running[server] = self.id
            sim.schedule_completion(self.id, server)
        # otherwise put the job into the queue
        else:
            sim.queues[server].append(self.id)
        # schedule the arrival of the next job
        sim.schedule_arrival(self.id+1)

class Completion(Event):
    def __init__(self, job_id, server):
        self.id = job_id  # currently unused, might be useful when extending
        self.server = server

    def process(self, sim: MMN):  # TODO: complete this method
        # print(f"Job {self.id} completed at time {sim.t}")
        assert sim.running[self.server] is not None
        # assert server_index is not None
        # set the completion time of the running job
        sim.completions[self.id] = sim.t
        # if the queue is not empty
        if len(sim.queues[self.server]) > 0:
            # get a job from the queue
            job = sim.queues[self.server].popleft()
            sim.running[self.server] = job
            # schedule its completion
            sim.schedule_completion(job, self.server)
        else:
            sim.running[self.server] = None

class Preemption(Event):
    def __init__(self, job_id, server, remaining_time):
        self.id = job_id  # currently unused, might be useful when extending
        self.server = server
        self.remaining_time = remaining_time

    def process(self, sim: MMN):  # TODO: complete this method
        servers = sample(range(sim.n), sim.d)  # sample d servers
        server = min(servers, key=lambda s: len(sim.queues[s])) # find the server with the shortest queue
        # if there is no running job, assign the incoming one and schedule its completion
        if sim.running[server] is None:
            sim.running[server] = self.id
            sim.schedule_completion(self.id, server, self.remaining_time)
        # otherwise put the job into the queue
        else:
            sim.queues[server].append(self.id)

        if len(sim.queues[self.server]) > 0:
            # get a job from the queue
            job = sim.queues[self.server].popleft()
            sim.running[self.server] = job
            # schedule its completion
            sim.schedule_completion(job, self.server)
        else:
            sim.running[self.server] = None

def run_simulation(lambd, mu, n, max_t, d):
    sim = MMN(lambd, mu, n, d)
    sim.run(max_t)
    completions = sim.completions
    W = (sum(completions.values()) - sum(sim.arrivals[job_id] for job_id in completions)) / len(completions)
    print(f"Average time spent in the system: lambd {lambd}: {W}")

    return sim.queueLengths

def run_simulation_wrapper(args):
    lambd, mu, n, max_t, d = args
    return run_simulation(lambd, mu, n, max_t, d)

def find_optimal_time_slice(lambd, mu, n, max_t, d, min_time_slice, max_time_slice, step):
    best_time_slice = min_time_slice
    best_metric = float('inf')

    for time_slice in range(min_time_slice, max_time_slice + 1, step):
        sim = MMN(lambd, mu, n, d, time_slice)
        sim.run(max_t)
        completions = sim.completions
        W = (sum(completions.values()) - sum(sim.arrivals[job_id] for job_id in completions)) / len(completions)
        if W < best_metric:
            best_metric = W
            best_time_slice = time_slice
        print(f"time_slice: {time_slice}, W: {W}")


    return best_time_slice

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambd', type=float, default=0.99)
    parser.add_argument('--mu', type=float, default=1)
    parser.add_argument('--max-t', type=float, default=10_000)
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--d', type=int, default=10)
    parser.add_argument('--csv', help="CSV file in which to store results")
    parser.add_argument("--seed", help="random seed")
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()

    if args.seed:
        seed(args.seed)  # set a seed to make experiments repeatable
    if args.verbose:
        logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)

    # sim = MMN(args.lambd, args.mu, args.n, args.d)
    # sim.run(args.max_t)

    # completions = sim.completions
    # W = (sum(completions.values()) - sum(sim.arrivals[job_id] for job_id in completions)) / len(completions)
    # print(f"Average time spent in the system: {W}")
    # # print(f"Theoretical expectation for waiting time: {args.lambd / (1 - args.lambd)}")
    # print(f"Theoretical expectation for random server choice: {1 / (1 - args.lambd)}")

    # optimal_time_slice = find_optimal_time_slice(args.lambd, args.mu,  args.n, args.max_t, args.d, 1, 100, 1)
    # print(f"Optimal time slice for lamda {args.lambd}: {optimal_time_slice}")

    # Create a list of parameters for each simulation
    simulations = [(lambd, args.mu, args.n, args.max_t, args.d) for lambd in [0.5, 0.9, 0.95, 0.99]]

    # Create a multiprocessing Pool
    with Pool() as p:
        results = p.map(run_simulation_wrapper, simulations)

    for lambd, queueLengths in zip([0.5, 0.9, 0.95, 0.99], results):
    # for lambd in [0.5, 0.9, 0.95, 0.99]:
        # queueLengths = run_simulation(lambd, args.mu, args.n, args.max_t, args.d)
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
    
    # if args.csv is not None:
    #     with open(args.csv, 'a', newline='') as f:
    #         writer = csv.writer(f)
    #         writer.writerow([args.lambd, args.mu, args.max_t, W])


if __name__ == '__main__':
    main()

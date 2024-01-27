#!/usr/bin/env python3

import argparse
import collections
import enum
import logging
import random

from matplotlib import pyplot as plt

from discrete_event_sim import Simulation, Event


class Condition(enum.Enum):
    """The condition of simulated individuals."""

    SUSCEPTIBLE = 0
    INFECTED = 1
    RECOVERED = 2


class SIR(Simulation):
    """The state of the simulation.

    We have the simulation parameters contact_rate and recovery_rate, plus the condition of every individual, as a
    list: conditions[i] represent the condition of the i-th individuals.

    s, i and r monitor the number of susceptible, infected and recovered individuals over time -- this is sampled
    periodically through the MonitorSIR event.
    """

    def __init__(self, population, infected, contact_rate, recovery_rate, plot_interval):
        super().__init__()  # call the initialization method from Simulation
        self.contact_rate = contact_rate
        self.recovery_rate = recovery_rate
        self.conditions = [Condition.SUSCEPTIBLE] * population  # a list of identical items of length 'population'
        for i in random.sample(range(population), infected):  # starting infected individuals
            self.infect(i)
        self.s, self.i, self.r = [], [], []  # values of susceptible, infected, recovered over time
        self.schedule(0, MonitorSIR(plot_interval))

    def schedule_contact(self, patient):
        """Schedule a patient's next contact."""

        other = random.randrange(len(self.conditions))  # choose a random contact
        self.schedule(random.expovariate(self.contact_rate), Contact(patient, other))

    def infect(self, i):
        """Patient i is infected."""

        self.log_info(f"{i} infected")
        self.conditions[i] = Condition.INFECTED
        self.schedule_contact(i)  # schedule the patient's next contact
        # (further contacts will be scheduled by the Contact event, see the process() function)
        self.schedule(random.expovariate(self.recovery_rate), Recover(i))  # schedule the patient's recovery


class Contact(Event):
    """A possible contagion event."""

    def __init__(self, source, destination):
        """Parameters: indexes of both the source and the destination of the possible contagion."""

        self.source = source
        self.destination = destination

    def process(self, sim):
        """If the patient is still infectious and the contact is susceptible, the latter will be infected."""

        sim.log_info(f"{self.source} contacts {self.destination}")
        if sim.conditions[self.source] != Condition.INFECTED:
            return  # healthy people can't infect
        if sim.conditions[self.destination] == Condition.SUSCEPTIBLE:
            sim.infect(self.destination)
        sim.schedule_contact(self.source)  # schedule the next contact


class Recover(Event):
    """A sick patient recovers."""

    def __init__(self, patient):
        self.patient = patient

    def process(self, sim):
        sim.log_info(f"{self.patient} recovered")
        sim.conditions[self.patient] = Condition.RECOVERED


class MonitorSIR(Event):
    """At any configurable interval, we save the number of susceptible, infected and recovered individuals."""

    def __init__(self, interval=1):
        self.interval = interval

    def process(self, sim):
        ctr = collections.Counter(sim.conditions)
        infected = ctr[Condition.INFECTED]
        sim.s.append(ctr[Condition.SUSCEPTIBLE])
        sim.i.append(infected)
        sim.r.append(ctr[Condition.RECOVERED])
        if infected > 0:  # if nobody is infected anymore, the simulation is over.
            sim.schedule(self.interval, self)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=int, default=1000)
    parser.add_argument("--infected", type=int, default=1, help="starting infected individuals")
    parser.add_argument("--seed", help="random seed")
    parser.add_argument("--avg-contact-time", type=float, default=1)
    parser.add_argument("--avg-recovery-time", type=float, default=3)
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--plot_interval", type=float, default=1, help="how often to collect data points for the plot")
    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)  # set a seed to make experiments repeatable
    if args.verbose:
        logging.basicConfig(format='{levelname}:{message}', level=logging.INFO, style='{')  # output info on stdout

    # the rates to use in random.expovariate are 1 over the desired mean
    sim = SIR(args.population, args.infected, 1 / args.avg_contact_time, 1 / args.avg_recovery_time, args.plot_interval)
    sim.run()
    assert all(c != Condition.INFECTED for c in sim.conditions)  # nobody should be infected at the end of the sim
    print(f"Simulation over at time {sim.t:.2f}")

    days = [i * args.plot_interval for i in range(len(sim.s))]  # compute the times at which values were taken
    plt.plot(days, sim.s, label="Susceptible")
    plt.plot(days, sim.i, label="Infected")
    plt.plot(days, sim.r, label="Recovered")
    plt.xlabel("Days")
    plt.ylabel("Individuals")
    plt.legend(loc=0)
    plt.grid()
    plt.show()


if __name__ == '__main__':  # run when this is run as a main file, not imported as a module
    main()

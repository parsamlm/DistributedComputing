import collections
import heapq
import logging

# TODO: implement the event queue!
# suggestion: have a look at the heapq library (https://docs.python.org/dev/library/heapq.html)
# and in particular heappush and heappop

class Simulation:
    """Subclass this to represent the simulation state.

    Here, self.t is the simulated time and self.events is the event queue.
    """

    def __init__(self):
        """Extend this method with the needed initialization.

        You can call super().__init__() there to call the code here.
        """

        self.t = 0  # simulated time
        # TODO: set up self.events as an empty queue
        self.events = []

    def schedule(self, delay, event):
        """Add an event to the event queue after the required delay."""

        # TODO: add event to the queue at time self.t + delay
        heapq.heappush(self.events,(self.t + delay, event))

    def run(self, max_t=float('inf')):
        """Run the simulation. If max_t is specified, stop it at that time."""

        while len(self.events) > 0:  # TODO: as long as the event queue is not empty:
            t, event = heapq.heappop(self.events) # TODO: get the first event from the queue
            if t > max_t:
                break
            self.t = t
            event.process(self)

    def log_info(self, msg):
        logging.info(f'{self.t:.2f}: {msg}')


class Event:
    """
    Subclass this to represent your events.

    You may need to define __init__ to set up all the necessary information.
    """

    def process(self, sim: Simulation):
        raise NotImplementedError

    def __lt__(self, other):
        """Method needed to break ties with events happening at the same time."""

        return id(self) < id(other)

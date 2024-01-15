import heapq
import logging


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
        self.events = []  # set up self.events as an empty queue, we have both Arrival and Completion objects here

    def schedule(self, delay, event):
        """Add an event to the event queue after the required delay."""

        # add both type of events into the heap, either arrival or completion, at time self.t + delay (random delay)
        heapq.heappush(self.events, (self.t + delay, event))

    def run(self, max_t=float('inf')):
        """Run the simulation. If max_t is specified, stop it at that time."""

        while self.events:  # as long as the event queue is not empty
            t, event = heapq.heappop(self.events)  # get the first event from event queue
            if t > max_t:
                break
            self.t = t
            event.process(self)
        print(f"It has finished at {self.t} seconds.")

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

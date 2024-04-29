from enum import Enum
from .simulation import Simulation
from .ship import Ship, Node
import random
import datetime

'''
Game class

'''
class Game:

    config = None
    state = None
    suite = None
    filename = None

    symbols = ['|', '/', '-', '\\']
    i = 0


    def __init__(self, config):
        self.config = config
        self.state = State.INITIALIZING
        self.i = 0
        self.data = []
        self.runs = 0
        self.fromFile = config["fromFile"]
        pass
        

    def update(self):
        match self.state:
            case State.INITIALIZING:
                self._handle_initializing()
            case State.READY:
                self._handle_ready()
                pass
            case State.RUNNING:
                self._handle_running()
                pass
            case State.TRANSITION:
                self._handle_transition()
                pass
    

    def change_state(self, new_state):
        self.state = new_state


    def _handle_initializing(self):
        print("initializing")
        print(self.config)
        print(self.suite)

        self.sims = [Simulation(self.config, Ship(fromFile=self.fromFile))]

        self.state = State.READY



    def _handle_ready(self):
        # print("ready")
        self.state = State.RUNNING


    def _handle_running(self):
        self.i = (self.i + 1) % len(self.symbols)
        print(f'\r\033[K{self.symbols[self.i]} running...', flush=True, end='')

        finished = True
        for sim in self.sims:
            sim.step()
            finished = finished and sim.finished
        
        if finished:
            self.state = State.TRANSITION
    

    # transition to the next set of simulations
    def _handle_transition(self):
        # print("transitioning to next layout")

        # update stats
        self.runs += 1
        self.data.append(self.sims[0].time)


        if self.runs == 10000:
            print("done")
            print("average time", sum(self.data) / len(self.data))
            self.state = State.DONE
            return

        fromFile = True
        if self.config["bot"] == "generalized":
            fromFile = False
        self.sims = [Simulation(self.config, Ship(fromFile=self.fromFile))]

        self.state = State.READY


class State(Enum):
    # sets up the game as per the config
    INITIALIZING  = 1
    # ready to start the game
    READY = 2
    # game is running
    RUNNING = 3
    # in the case of certain configurations, the game may need to modify its configuration
    TRANSITION = 4
    # game is done
    DONE = 5
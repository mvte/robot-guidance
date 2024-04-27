from .ship import Ship, Node
from .bot_factory import botFactory
from .agents.crewmate import Crewmate
import math
import random

class Simulation:
    # config
    ship = None
    bot = None
    crew = None
    finished = False

    # state
    time = 0

    def __init__(self, config, ship, botPos = None, crewmatePos = None):
        self.config = config
        self.ship = ship
        self.bot = None
        self.crew = None
        botPos = (config["botPos"]["x"], config["botPos"]["y"])
        crewmatePos = (config["crewPos"]["x"], config["crewPos"]["y"])

        if config["bot"] != "none":
            self._placeBot(config["bot"], botPos)

        self._placeCrew(crewmatePos)

    
    # places the bot in the given position, or a random open position if no position is given
    def _placeBot(self, whichBot, pos):
        bot = botFactory(whichBot, pos, self.ship)
        self.bot = bot


    # places the crewmate in the given position, or a random open position if no position is given
    def _placeCrew(self, crewPos):
        if crewPos != (-1, -1):
            self.crew = Crewmate(crewPos)
        else:
            while True:
                x = random.randint(0, self.ship.dim - 1)
                y = random.randint(0, self.ship.dim - 1)
                if self.ship.board[x][y] == Node.OPEN:
                    self.crew = Crewmate((x, y))
                    break

        self.initialCrewPos = self.crew.pos
            

    def _getManhattanDistance(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        return abs(x1 - x2) + abs(y1 - y2)


    def step(self):
        if self.finished:
            return   
        
        self.time += 1

        # move the bot
        if self.bot:
            self.bot.pos = self.bot.computeNextStep(self.ship, self.crew.pos)
            

        # move the crewmate
        if self.bot:
            self.crew.pos = self.crew.computeNextStep(self.ship, self.bot.pos)
        else:
            self.crew.pos = self.crew.computeNextStep(self.ship, None)

        # check if the crewmate is on the teleport pad
        if self.ship.board[self.crew.pos[0]][self.crew.pos[1]] == Node.TP:
            self.endSimulation()
            return
        

    def endRun(self, success):
        pass


    def endSimulation(self):
        self.finished = True
        # print("\nsimulation has ended")
        # print("initial crew pos:", self.initialCrewPos)
        # print("num steps:", self.time)
        

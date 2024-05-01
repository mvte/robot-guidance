from .ship import Ship, Node
from .bot_factory import botFactory
from .agents.crewmate import Crewmate
import math
import random

class Simulation:
    def __init__(self, config, ship, botPos = None, crewmatePos = None):
        self.config = config
        self.ship = ship
        self.bot = None
        self.crew = None

        self.time = 0
        self.finished = False
        self.rewards = []

        botPos = (config["botPos"]["x"], config["botPos"]["y"])
        crewmatePos = (config["crewPos"]["x"], config["crewPos"]["y"])

        self._placeCrew(crewmatePos)

        if config["bot"] != "none":
            self._placeBot(config["bot"], botPos)

    
    # places the bot in the given position, or a random open position if no position is given
    def _placeBot(self, whichBot, pos):
        if pos == (-1, -1):
            while True:
                x = random.randint(0, self.ship.dim - 1)
                y = random.randint(0, self.ship.dim - 1)
                if self.ship.board[x][y] == Node.OPEN and self.crew.pos != (x, y):
                    pos = (x, y)
                    break      

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

        old_bot_pos = self.bot.pos
        old_crew_pos = self.crew.pos

        # move the bot
        if self.bot:
            self.bot.pos = self.bot.computeNextStep(self.ship, self.crew.pos)
            

        # move the crewmate
        if self.bot:
            self.crew.pos = self.crew.computeNextStep(self.ship, self.bot.pos)
        else:
            self.crew.pos = self.crew.computeNextStep(self.ship, None)

        # calculate reward
        reward = self.calculate_reward(old_bot_pos, old_crew_pos, self.bot.pos, self.crew.pos)
        self.rewards.append(reward)

        # check if the crewmate is on the teleport pad
        if self.ship.board[self.crew.pos[0]][self.crew.pos[1]] == Node.TP:
            self.endSimulation()
            return
        

    def endRun(self, success):
        pass


    def endSimulation(self):
        self.finished = True
        # print(sum(self.rewards) / len(self.rewards))
        # print("\nsimulation has ended")
        # print("initial crew pos:", self.initialCrewPos)
        # print("num steps:", self.time)
        
    # def calculate_reward(self, old_bot_pos, old_crew_pos, bot_pos, crew_pos):
    #     reward = 0.0

    #     # positive reward for reaching the teleport pad
    #     if crew_pos == (5, 5):
    #         return 100.0

    #     # negative reward for taking long
    #     reward -= 1

    #     # positive reward if the crew is closer to the teleport pad (small since this may occur by random chance)
    #     crew_dist = abs(crew_pos[0] - 5) + abs(crew_pos[1] - 5)
    #     old_crew_dist = abs(old_crew_pos[0] - 5) + abs(old_crew_pos[1] - 5)
    #     if crew_dist < old_crew_dist:
    #         reward += 1

    #     # positive reward if the bot is closer to the crew
    #     old_bot_dist_to_crew = abs(old_bot_pos[0] - crew_pos[0]) + abs(old_bot_pos[1] - crew_pos[1])
    #     bot_dist_to_crew = abs(bot_pos[0] - crew_pos[0]) + abs(bot_pos[1] - crew_pos[1])
    #     if bot_dist_to_crew < old_bot_dist_to_crew:
    #         reward += 2
    #     else:
    #         reward -= 3

    #     # positive reward if bot positions itself such that the crewmate is between the bot and the teleport pad when adjacent
    #     is_adj = abs(bot_pos[0] - crew_pos[0]) + abs(bot_pos[1] - crew_pos[1]) == 1
    #     bot_dist = abs(bot_pos[0] - 5) + abs(bot_pos[1] - 5)
    #     if is_adj and bot_dist > crew_dist:
    #         reward += 5
    #     if is_adj and bot_dist < crew_dist:
    #         reward -= 7

    #     # reward if bot is adjacent
    #     if is_adj:
    #         reward += 3
    #     else:
    #         reward -= 2

    #     return reward


    def calculate_reward(sim, old_bot_pos, old_crew_pos, bot_pos, crew_pos):
        reward = 0.0

        # positive reward for reaching the teleport pad
        if crew_pos == (5, 5):
            return 100.0
        
        # encourage being adjacent to the crewmate
        adj = abs(bot_pos[0] - crew_pos[0]) + abs(bot_pos[1] - crew_pos[1]) == 1
        if adj:
            reward += 5

        # encourage actions that push the crewmate closer to the teleport pad
        close = abs(bot_pos[0] - crew_pos[0]) < 3 and abs(bot_pos[1] - crew_pos[1]) < 3
        crew_dist = abs(crew_pos[0] - 5) + abs(crew_pos[1] - 5)
        old_crew_dist = abs(old_crew_pos[0] - 5) + abs(old_crew_pos[1] - 5)
        if close and crew_dist < old_crew_dist:
            reward += 10 * (1/crew_dist)
        else:
            reward -= 0.5 * old_crew_dist
        
        # # encourage moving towards the crewmate, and discourage moving away
        # old_dist = abs(old_bot_pos[0] - old_crew_pos[0]) + abs(old_bot_pos[1] - old_crew_pos[1])
        # new_dist = abs(bot_pos[0] - old_crew_pos[0]) + abs(bot_pos[1] - old_crew_pos[1])
        # if new_dist < old_dist:
        #     reward += 15
        # else:
        #     reward -= 5
        
        # # encourage placing the crewmate between the bot and the teleport pad, and discourage incorrect positioning
        # close = abs(bot_pos[0] - crew_pos[0]) < 2 and abs(bot_pos[1] - crew_pos[1]) < 2
        # if adj or close:
        #     bot_dist = abs(bot_pos[0] - 5) + abs(bot_pos[1] - 5)
        #     if crew_dist < bot_dist:
        #         reward += 30
        #     else:
        #         reward -= 10

        return reward
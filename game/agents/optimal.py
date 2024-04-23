import numpy as np
from game.calc.with_bot import PolicyIteration

class OptimalBot:
    def __init__(self, pos, ship):
        self.pos = pos

        self.polyIter = PolicyIteration(ship.board)
        self.policy = np.load("policy.npy")

    
    def computeNextStep(self, ship, crewPos):
        crew_index = self.polyIter.open_cells[crewPos]
        action_index = self.policy[self.polyIter.open_cells[self.pos]][crew_index]
        action = self.polyIter.action_space[int(action_index)]
        return (self.pos[0] + action[0], self.pos[1] + action[1])


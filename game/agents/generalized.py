import numpy as np
import torch
from game.ship import Node
from game.calc.with_bot import PolicyIteration
from game.calc.learned import BotNetwork

class GeneralizedBot():
    def __init__(self, pos, ship):
        self.pos = pos

        self.polyIter = PolicyIteration(ship.board)

        self.model = BotNetwork()
        self.model.load_state_dict(torch.load("data/generalized_bot_chk.pth"))
        self.model
        self.shipTensor = torch.zeros(121)
        for i in range(11):
            for j in range(11):
                if ship.board[i][j] == Node.OPEN:
                    self.shipTensor[i * 11 + j] = 1

    def computeNextStep(self, ship, crewPos):
        posTensor = torch.zeros(121)
        crewTensor = torch.zeros(121)

        posTensor[self.pos[0] * 11 + self.pos[1]] = 1
        crewTensor[crewPos[0] * 11 + crewPos[1]] = 1

        validMoveTensor = torch.zeros(9)
        validMoves = self.polyIter.compute_bot_actions(self.pos, crewPos)

        for i, action in enumerate(BotNetwork.ACTION_SPACE):
            if action in validMoves:
                validMoveTensor[i] = 1

        with torch.no_grad():
            output = self.model(posTensor[None,:], crewTensor[None,:], self.shipTensor[None,:], validMoveTensor[None,:])
            action_idx = output.argmax(dim=1)

        action = BotNetwork.ACTION_SPACE[action_idx.item()]
        return (self.pos[0] + action[0], self.pos[1] + action[1])
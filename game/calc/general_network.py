import torch
import torch.nn as nn
from torch.distributions import Categorical

class GeneralNetwork(nn.Module):
    ACTION_SPACE = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    def __init__(self):
        super(GeneralNetwork, self).__init__()

        self.botEncoder = nn.Linear(121, 64)
        self.crewEncoder = nn.Linear(121, 64)
        self.shipEncoder = nn.Linear(121, 64)

        self.fc1 = nn.Linear(192, 256)
        self.fc2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 9)


    def forward(self, bot, crew, ship, valid_moves):
        assert not torch.isnan(bot).any()
        assert not torch.isnan(crew).any()
        assert not torch.isnan(ship).any()
        assert not torch.isnan(valid_moves).any()

        botEncoded = self.botEncoder(bot)
        crewEncoded = self.crewEncoder(crew)
        shipEncoded = self.shipEncoder(ship)

        x = torch.cat((botEncoded, crewEncoded, shipEncoded), dim=1)
        x = torch.tanh(x)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))

        x = self.output(x)
        # send invalid moves to neg inf (or close enough to it)
        x[valid_moves == 0] = -1e6

        return x
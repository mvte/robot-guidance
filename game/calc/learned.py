import torch
import torch.nn as nn

class BotNetwork(nn.Module):
    ACTION_SPACE = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    def __init__(self):
        super(BotNetwork, self).__init__()

        self.botEncoder = nn.Linear(121, 64)
        self.crewEncoder = nn.Linear(121, 64)
        self.shipEncoder = nn.Linear(121, 64)

        self.fc1 = nn.Linear(192, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 9)


    def forward(self, bot, crew, ship):
        botEncoded = self.botEncoder(bot)
        crewEncoded = self.crewEncoder(crew)
        shipEncoded = self.shipEncoder(ship)

        x = torch.cat((botEncoded, crewEncoded, shipEncoded), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.validMoveMask(self.output(x), bot, crew, ship)

        return x
    

    def validMoveMask(self, x, bot, crew, ship):
        # mask invalid moves
        for i in range(9):
            if i not in self.computeBotActions(bot, crew, ship):
                x[0][i] = -float('inf')

        return x
    

    def computeBotActions(self, bot, crew, ship):
        # convert bot vector to tuple
        bot = torch.argmax(bot).item()
        bot = (bot // 11, bot % 11)

        # convert crew vector to tuple
        crew = torch.argmax(crew).item()
        crew = (crew // 11, crew % 11)

        # convert ship vector to 2d array
        ship = ship.view(11, 11)

        # compute valid actions
        validActions = set()
        for i, action in enumerate(self.ACTION_SPACE):
            x, y = bot
            x += action[0]
            y += action[1]

            if x < 0 or x >= 11 or y < 0 or y >= 11:
                continue

            if ship[x][y] == 1:
                validActions.add(i)
        
        return validActions
        

def train():
    pass


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from game.ship import Ship, Node
from game.calc.with_bot import PolicyIteration

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
        x = torch.relu(x)
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
        batch_size = bot.size(0)
        validActions = [set() for _ in range(batch_size)]

        # convert bot vector to tuple
        bot_positions = torch.argmax(bot, dim=1)
        bot_positions = [(pos // 11, pos % 11) for pos in bot_positions]

        # convert crew vector to tuple
        crew_positions = torch.argmax(crew, dim=1)
        crew_positions = [(pos // 11, pos % 11) for pos in crew_positions]

        # convert ship vector to 2d array
        ship = ship.view(batch_size, 11, 11)

        # compute valid actions
        for i in range(batch_size):
            bot_pos = bot_positions[i]
            crew_pos = crew_positions[i]
            ship_map = ship[i]

            for j, action in enumerate(self.ACTION_SPACE):
                x, y = bot_pos
                x += action[0]
                y += action[1]

                if 0 <= x < 11 and 0 <= y < 11 and ship_map[x][y] == 1 and (x, y) != crew_pos:
                    validActions[i].add(j)
        
        return validActions
        

def train():
    # generate data set
    dataset = generate_data()

    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create network
    net = BotNetwork().to(device)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # training loop
    num_epochs = 100
    batch_size = 32

    for epoch in range(num_epochs):
        for i in range(0, len(dataset), batch_size):
            bot, crew, ship, action = zip(*dataset[i:i+batch_size])
            bot = torch.stack(bot).to(device)
            crew = torch.stack(crew).to(device)
            ship = torch.stack(ship).to(device)
            action = torch.tensor(action).type(torch.LongTensor).to(device)

            # forward pass
            output = net(bot, crew, ship)

            # compute loss
            loss = criterion(output, action)

            # backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"epoch {epoch + 1}/{num_epochs}, loss: {loss.item()}")

    # save the model
    net = net.to("cpu")
    torch.save(net.state_dict(), "bot.pth")


def generate_data():
    ship = Ship(fromFile=True)
    board = ship.board
    polyIter = PolicyIteration(ship.board)
    policy = np.load("policy.npy")

    bot_positions = [(i, j) for i in range(11) for j in range(11)]
    crew_positions = [(i, j) for i in range(11) for j in range(11)]
    ship = torch.zeros(11, 11)

    # place 1s where ship cells are open
    for i in range(11):
        for j in range(11):
            if board[i][j] == Node.OPEN:
                ship[i][j] = 1
    ship = ship.flatten()

    # get optimal actions for each cell pair
    dataset = []
    for bot_pos in bot_positions:
        for crew_pos in crew_positions:
            if bot_pos == crew_pos:
                continue
            if bot_pos not in polyIter.open_cells or crew_pos not in polyIter.open_cells:
                continue

            bot_index = polyIter.open_cells[bot_pos]
            crew_index = polyIter.open_cells[crew_pos]

            bot = torch.zeros(121)
            crew = torch.zeros(121)
            bot[bot_pos[0] * 11 + bot_pos[1]] = 1
            crew[crew_pos[0] * 11 + crew_pos[1]] = 1

            optimal_action = policy[bot_index][crew_index]

            # add data to dataset
            dataset.append((bot, crew, ship, optimal_action))

    return dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

from game.ship import Ship, Node
from game.simulation import Simulation
from game.calc.with_bot import PolicyIteration

class GeneralNetwork(nn.Module):
    ACTION_SPACE = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    def __init__(self):
        super(GeneralNetwork, self).__init__()

        self.botEncoder = nn.Linear(121, 64)
        self.crewEncoder = nn.Linear(121, 64)
        self.shipEncoder = nn.Linear(121, 64)

        self.fc1 = nn.Linear(192, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 9)


    def forward(self, bot, crew, ship, valid_moves):
        botEncoded = self.botEncoder(bot)
        if torch.isnan(botEncoded).any():
            print("botEncoded has nan")
            print(bot)
            print(botEncoded)
            print(self.botEncoder.weight)
            exit(1)
        crewEncoded = self.crewEncoder(crew)
        if torch.isnan(crewEncoded).any():
            print("crewEncoded has nan")
            print(crew)
            print(crewEncoded)
            print(self.crewEncoder.weight)
            exit(1)
        shipEncoded = self.shipEncoder(ship)
        if torch.isnan(shipEncoded).any():
            print("shipEncoded has nan")
            print(ship)
            print(shipEncoded)
            print(self.shipEncoder.weight)
            exit(1)

        x = torch.cat((botEncoded, crewEncoded, shipEncoded), dim=1)
        x = torch.relu(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        x = self.output(x)
        # send invalid moves to neg inf (or close enough to it)
        x[valid_moves == 0] = -1e6

        softmax_x = torch.softmax(x, dim=1)
        try :
            dist = Categorical(softmax_x)
        except Exception as e:
            print("Error in forward pass")
            print(x)
            print(softmax_x)
            print(e)
            exit(1)

        return dist


class CriticNetwork(nn.Module):
    def __init__(self):
        super(CriticNetwork, self).__init__()

        self.botEncoder = nn.Linear(121, 64)
        self.crewEncoder = nn.Linear(121, 64)
        self.shipEncoder = nn.Linear(121, 64)

        self.fc1 = nn.Linear(192, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)


    def forward(self, bot, crew, ship):
        botEncoded = self.botEncoder(bot)
        crewEncoded = self.crewEncoder(crew)
        shipEncoded = self.shipEncoder(ship)

        x = torch.cat((botEncoded, crewEncoded, shipEncoded), dim=1)
        x = torch.relu(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        x = self.output(x)

        return x


class Memory():
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.is_terminals = []
        
    def add(self, state, action, logprob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.values.append(value)
        self.is_terminals.append(done)

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.values[:]
        del self.is_terminals[:]


def learn():
    # initialize config to force the simulation to use random spawn points
    config = {
        "bot": "controlled",
        "botPos": {
            "x": -1,
            "y": -1
        },
        "crewPos": {
            "x": -1,
            "y": -1
        },
    }

    # hyperparameters
    gamma = 0.99
    max_steps = 50
    num_epochs = 1000
    epsilon = 0.2
    update_epochs = 5
    alpha = 0.0003
    gae_lambda = 0.95

    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create networks
    net = GeneralNetwork().to(device)
    critic = CriticNetwork().to(device)

    # define optimizer 
    optimizer = optim.Adam(net.parameters(), lr=alpha)
    critic_optimizer = optim.Adam(critic.parameters(), lr=alpha)
    
    # create memory 
    memory = Memory()

    for epoch in range(num_epochs):
        # collect data on various ship layouts
        for _ in range(5):
            play(config, device, net, critic, memory, max_steps)

        # get data from memory
        bot, crew, ship, valid_moves = zip(*memory.states)
        bot = torch.stack(bot).to(device)
        crew = torch.stack(crew).to(device)
        ship = torch.stack(ship).to(device)
        valid_moves = torch.tensor(valid_moves).to(device)
        logprobs = torch.stack(memory.logprobs).to(device)
        rewards = torch.tensor(memory.rewards).to(device)
        values = torch.tensor(memory.values).to(device) 
        is_terminals = torch.tensor(memory.is_terminals).to(device)

        # print some stats
        print(f"avg reward: {values.mean().item()}")

        # compute advantages
        advantages = torch.zeros_like(rewards).to(device)
        for t in range(len(rewards) - 1):
            discount = 1
            a_t = 0
            for k in range(t, len(rewards) - 1):
                a_t += discount * (rewards[k] + gamma * values[k + 1] * (1 - is_terminals[k]) - values[k])
                discount *= gamma * gae_lambda
            advantages[t] = a_t
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        # descend the policy gradient
        for _ in range(update_epochs):
            # forward pass (actor)
            dist = net(bot, crew, ship, valid_moves)
            logprobs_new = dist.log_prob(dist.sample())

            # forward pass (critic)
            values_pred = critic(bot, crew, ship).squeeze()       

            # compute loss for actor (ppo clip loss)
            ratio = torch.exp(logprobs_new - logprobs)
            clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
            loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

            # compute loss for critic
            returns = advantages + values
            critic_loss = nn.MSELoss()(values_pred, returns)

            total_loss = loss + 0.5 * critic_loss

            # backward pass and optimization
            optimizer.zero_grad()
            critic_optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
            optimizer.step()
            critic_optimizer.step()

        print(f"epoch {epoch + 1}/{num_epochs}, loss: {loss.item()}")

        # checkpoint
        if (epoch + 1) % 100 == 0:
            print("checkpointing")
            torch.save(net.state_dict(), "data/generalized_bot_chk.pth")
            torch.save(critic.state_dict(), "data/critic_chk.pth")
        memory.clear()

    # save the model
    net = net.to("cpu")
    torch.save(net.state_dict(), "generalized_bot.pth")


def calculate_reward(sim, bot_pos, crew_pos):
    reward = 0

    # positive reward for reaching the teleport pad
    if crew_pos == (5, 5):
        reward += 100

    # negative reward for taking a step
    reward -= 1

    # negative reward if the crewmate is far from the teleport pad
    reward -= sim._getManhattanDistance(crew_pos, (5, 5))

    # reward for maintaining adjacency to the crewmate
    if abs(bot_pos[0] - crew_pos[0]) + abs(bot_pos[1] - crew_pos[1]) == 1:
        reward += 5

    return float(reward)


def play(config, device, net, critic, memory, max_steps=50):
    # create simulation
    sim = Simulation(config, Ship())

    ship = torch.zeros(121).to(device)
    for i in range(11):
            for j in range(11):
                if sim.ship.board[i][j] == Node.OPEN:
                    ship[i * 11 + j] = 1
    polyIter = PolicyIteration(sim.ship.board)

    # collect data from simulation
    while not sim.finished and sim.time < max_steps:
        # extract state from simulation
        bot = torch.zeros(121).to(device)
        crew = torch.zeros(121).to(device)
        bot[sim.bot.pos[0] * 11 + sim.bot.pos[1]] = 1
        crew[sim.crew.pos[0] * 11 + sim.crew.pos[1]] = 1

        botActions = polyIter.compute_bot_actions(sim.bot.pos, sim.crew.pos)
        valid_moves = [1 if action in botActions else 0 for action in GeneralNetwork.ACTION_SPACE]
        valid_moves_tensor = torch.tensor(valid_moves).to(device)

        # get action
        with torch.no_grad():
            dist = net(bot[None,:], crew[None,:], ship[None,:], valid_moves_tensor[None,:])
            action = dist.sample()
            logprob = dist.log_prob(action)

        # get value
        with torch.no_grad():
            value = critic(bot[None,:], crew[None,:], ship[None,:])

        # take action
        action = action.item()
        new_pos = (sim.bot.pos[0] + GeneralNetwork.ACTION_SPACE[action][0], sim.bot.pos[1] + GeneralNetwork.ACTION_SPACE[action][1])
        sim.bot.pos = new_pos

        # calculate reward
        reward = calculate_reward(sim, sim.bot.pos, sim.crew.pos)

        # add to memory
        memory.add((bot, crew, ship, valid_moves), action, logprob, reward, value, 1 if sim.finished else 0)

        # update simulation
        sim.step()
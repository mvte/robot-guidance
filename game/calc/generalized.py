import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
        assert not torch.isnan(bot).any()
        assert not torch.isnan(crew).any()
        assert not torch.isnan(ship).any()
        assert not torch.isnan(valid_moves).any()

        botEncoded = self.botEncoder(bot)
        crewEncoded = self.crewEncoder(crew)
        shipEncoded = self.shipEncoder(ship)

        x = torch.cat((botEncoded, crewEncoded, shipEncoded), dim=1)
        x = torch.relu(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        x = self.output(x)
        # send invalid moves to neg inf (or close enough to it)
        x[valid_moves == 0] = -1e6

        return x


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
        self.ep_indices = [0]
        
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




def learn(load=False):
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


    # create reward csv to save reward
    with open("data/reward.csv", "w") as f:
        f.write("reward\n")

    # hyperparameters
    gamma = 0.99
    max_steps = 200
    num_epochs = 10000
    epsilon = 0.2
    update_epochs = 3
    num_envs = 20
    lr_actor = 0.0003
    lr_critic = 0.001

    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create networks
    net = GeneralNetwork().to(device)
    critic = CriticNetwork().to(device)
    if load:
        print("loading model")
        net.load_state_dict(torch.load("data/generalized_bot_chk.pth"))
        critic.load_state_dict(torch.load("data/critic_chk.pth"))
    else:
        print("training from scratch")
        torch.save(net.state_dict(), "data/generalized_bot_chk.pth")
        torch.save(critic.state_dict(), "data/critic_chk.pth")  
        

    # define optimizer 
    optimizer = optim.Adam(net.parameters(), lr=lr_actor, eps=1e-5)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr_critic, eps=1e-5)
    
    # create memory 
    memory = Memory()

    for epoch in range(num_epochs):
        # collect data on various ship layouts
        for _ in range(num_envs):
            play(config, device, net, critic, memory, max_steps)
            memory.ep_indices.append(len(memory.states))

        # get data from memory
        rewards = torch.tensor(memory.rewards).to(device)
        is_terminals = torch.tensor(memory.is_terminals).to(device)
        values = torch.tensor(memory.values).to(device)

        # print some stats
        print(f"avg reward: {rewards.mean().item()}")
        with open("data/reward.csv", "a") as f:
            f.write(f"{rewards.mean().item()}\n")


        # per episode
        advantages_all = []
        returns_all = []
        for i in range(len(memory.ep_indices) - 1):
            ep_start = memory.ep_indices[i]
            ep_end = memory.ep_indices[i+1]

            # compute returns
            returns = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(rewards[ep_start:ep_end]), reversed(is_terminals[ep_start:ep_end])):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + gamma * discounted_reward
                returns.insert(0, discounted_reward)
            returns = torch.tensor(returns).to(device)
            returns_all.append(returns)
            
            # compute advantages
            advantages = returns.detach() - values[ep_start:ep_end].detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
            advantages_all.append(advantages)
        advantages_all = torch.cat(advantages_all)

        # print some stats
        # returns_all = torch.cat(returns_all)
        # print("mean episodic return", returns_all.mean().item())

        minibatches = create_minibatches(memory, advantages_all)

        # ascend the policy gradient
        for i in range(update_epochs):
            for j, batch in enumerate(minibatches):
                bot, crew, ship, valid_moves, action, logprobs, rewards, values, advantages = batch

                # forward pass (actor)
                output = net(bot, crew, ship, valid_moves)
                assert not torch.isnan(output).any()
                dist = Categorical(logits=output)
                logprobs_new = dist.log_prob(action)
                entropy = dist.entropy().mean()

                # forward pass (critic)
                values_pred = critic(bot, crew, ship).squeeze()      

                # compute loss for actor (ppo clip loss)
                ratios = torch.exp(logprobs_new - logprobs.squeeze())
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages
                loss = (-torch.min(surr1, surr2)).mean()

                # # debug variables
                # with torch.no_grad():
                #     if j == 0 and i == 0:
                #         kl = ((ratios.detach() - 1) - (logprobs_new.detach() - logprobs.detach().squeeze())).mean().item()
                #         print(f"kl: {kl}")
                #         print(f"entropy: {entropy.item()}")

                # compute loss for critic
                returns = advantages - values
                critic_loss = nn.MSELoss()(values_pred, returns.squeeze())

                total_loss = loss + 0.5 * critic_loss - 0.01 * entropy

                # optimize
                optimizer.zero_grad()
                critic_optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                critic_optimizer.step()

                for name, param in net.named_parameters():
                    if torch.isnan(param).any():
                        print(f'Parameter {name} has NaN value!')



        print(f"epoch {epoch + 1}/{num_epochs}, actor loss: {loss.item()}, critic loss: {critic_loss.item()}")
        # checkpoint
        if (epoch + 1) % 25 == 0:
            print("checkpointing")
            torch.save(net.state_dict(), "data/generalized_bot_chk.pth")
            torch.save(critic.state_dict(), "data/critic_chk.pth")

        memory.clear()

    # save the model
    net = net.to("cpu")
    torch.save(net.state_dict(), "generalized_bot.pth")


def create_minibatches(memory, advantages_all, batch_size=512):
    bot, crew, ship, valid_moves = zip(*memory.states)
    experiences = list(zip(bot, crew, ship, valid_moves, memory.actions, memory.logprobs, memory.rewards, memory.values, advantages_all))
    np.random.shuffle(experiences)

    minibatches = []
    for i in range(0, len(experiences), batch_size):
        batch = experiences[i:i+batch_size]
        bot, crew, ship, valid_moves, action, logprobs, rewards, values, advantages = zip(*batch)
        bot = torch.stack(bot).to("cuda")
        crew = torch.stack(crew).to("cuda")
        ship = torch.stack(ship).to("cuda")
        valid_moves = torch.tensor(valid_moves).to("cuda")
        action = torch.tensor(action).to("cuda")
        logprobs = torch.stack(logprobs).to("cuda")
        rewards = torch.tensor(rewards).to("cuda")
        values = torch.tensor(values).to("cuda")
        advantages = torch.tensor(advantages).to("cuda")

        minibatches.append((bot, crew, ship, valid_moves, action, logprobs, rewards, values, advantages))

    return minibatches


@torch.no_grad()
def calculate_reward(sim, old_bot_pos, old_crew_pos, bot_pos, crew_pos):
    reward = 0.0

    # positive reward for reaching the teleport pad
    if crew_pos == (5, 5):
        return 500.0
    
    
    # # negative reward for being far from the crewmate
    # reward -= (abs(bot_pos[0] - crew_pos[0]) + abs(bot_pos[1] - crew_pos[1]))


    # positive reward for the crewmate being close to the teleport pad 
    crew_dist = abs(crew_pos[0] - 5) + abs(crew_pos[1] - 5)
    reward += 1 / crew_dist * 5
    
    # # encourage being adjacent to the crewmate
    # adj = abs(bot_pos[0] - crew_pos[0]) + abs(bot_pos[1] - crew_pos[1]) == 1
    # if adj:
    #     reward += 10
    
    # # encourage moving towards the crewmate, and discourage moving away
    # old_dist = abs(old_bot_pos[0] - old_crew_pos[0]) + abs(old_bot_pos[1] - old_crew_pos[1])
    # new_dist = abs(bot_pos[0] - old_crew_pos[0]) + abs(bot_pos[1] - old_crew_pos[1])
    # if new_dist < old_dist:
    #     reward += 20
    # else:
    #     reward -= 10
    
    # # encourage placing the crewmate between the bot and the teleport pad, and discourage incorrect positioning
    # close = abs(bot_pos[0] - crew_pos[0]) < 2 and abs(bot_pos[1] - crew_pos[1]) < 2
    # if adj or close:
    #     bot_dist = abs(bot_pos[0] - 5) + abs(bot_pos[1] - 5)
    #     if crew_dist < bot_dist:
    #         reward += 30
    #     else:
    #         reward -= 25

    return reward

@torch.no_grad()
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
    old_crew_pos = sim.crew.pos
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
        output = net(bot[None,:], crew[None,:], ship[None,:], valid_moves_tensor[None,:])
        dist = Categorical(logits=output)
        action = dist.sample()
        logprob = dist.log_prob(action)

        # get value
        value = critic(bot[None,:], crew[None,:], ship[None,:])

        # take action
        old_bot_pos = sim.bot.pos
        old_crew_pos = sim.crew.pos
        action = action.item()
        new_pos = (sim.bot.pos[0] + GeneralNetwork.ACTION_SPACE[action][0], sim.bot.pos[1] + GeneralNetwork.ACTION_SPACE[action][1])
        sim.bot.pos = new_pos
        sim.step()

        # calculate reward
        reward = calculate_reward(sim, old_bot_pos, old_crew_pos, sim.bot.pos, sim.crew.pos)

        # add to memory
        memory.add((bot, crew, ship, valid_moves), action, logprob, reward, value, 1 if sim.finished else 0)

        # update simulation
        old_crew_pos = sim.crew.pos



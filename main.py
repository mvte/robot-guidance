from game.game import Game, State
from visual import Visual
import sys
import json
import datetime

from game.ship import Ship, printBoard
from game.calc.no_bot import t_no_bot, uev
from game.calc.with_bot import PolicyIteration, improvement, minimal_improvement_config, minimal_average_value

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    timestart = datetime.datetime.now()

    # default to game
    if len(sys.argv) < 2:
        game()
        print("time elapsed", datetime.datetime.now() - timestart)
        print("game over")


    match sys.argv[1]:
        case "game":
            game()
        case "tnb":
            calc_t_no_bot()
        case "tb":
            use_existing = "use_existing" in sys.argv
            calc_t_bot(use_existing)
        case "train":
            train()
        case "show_data":
            show_data()
        case "show_policy":
            crewPos = (int(sys.argv[2]), int(sys.argv[3]))
            show_policy(crewPos)
        case "show_values":
            botPos = (int(sys.argv[2]), int(sys.argv[3]))
            show_values(botPos)
        case _:
            print("invalid command")

    
    print("time elapsed", datetime.datetime.now() - timestart)
    print("game over")


def game():
    config = None
    with open("config.json", "r") as f:
        config = json.load(f)
    game = Game(config)

    visual = None
    if len(sys.argv) > 1 and sys.argv[1] == "visualize":
        visual = Visual(game)
    if len(sys.argv) > 2 and sys.argv[2] == "visualize":
        visual = Visual(game)

    while True:
        game.update()
        
        if visual:
            visual.update()

        if game.state == State.DONE:
            break


def calc_t_no_bot():
    ship = Ship(fromFile = True)
    printBoard(ship.board)

    expected_time = t_no_bot(ship.board)
    np.save("t_no_bot.npy", expected_time)

    print(uev(expected_time))
    sns.heatmap(expected_time, vmax=300, annot=True, fmt=".2f")
    plt.show()

    
def calc_t_bot(use_existing):
    ship = Ship(fromFile = True)
    printBoard(ship.board)
        
    if use_existing:
        values = np.load(f"values.npy")
        policy = np.load(f"policy.npy")
        polyIter = PolicyIteration(ship.board, values, policy)
    else:
        polyIter = PolicyIteration(ship.board)

    values, policy = polyIter.policy_iteration()

    # save the values and policy matrices
    np.save(f"values.npy", values)
    np.save(f"policy.npy", policy)

    # fix one bot position, and reshape the values array, and plot it
    botPos = (0, 0)
    bot_index = polyIter.open_cells[botPos]
    values = values[bot_index]
    reshaped = np.full((len(ship.board), len(ship.board)), np.nan)

    for cell, index in polyIter.open_cells.items():
        i, j = cell
        reshaped[i][j] = values[index]

    sns.heatmap(reshaped, annot=True, fmt=".2f")
    plt.show()


def show_data():
    ship = Ship(fromFile = True)
    printBoard(ship.board)
    polyIter = PolicyIteration(ship.board)

    values = np.load(f"values.npy")
    policy = np.load(f"policy.npy")

    # average improvement
    no_bot = np.load("t_no_bot.npy")
    improvement(no_bot, values, polyIter)

    # configuration with minimal improvement
    minimal_improvement_config(no_bot, values, polyIter)

    # position with minimal average time to escape
    minimal_average_value(values, polyIter)


def show_policy(crewPos):
    ship = Ship(fromFile = True)
    printBoard(ship.board)
    polyIter = PolicyIteration(ship.board)

    policy = np.load(f"policy.npy")
    crew_index = polyIter.open_cells[crewPos]
    policy = policy[:,crew_index]
    reshaped_policy = np.full((len(ship.board), len(ship.board)), np.nan)

    for cell, index in polyIter.open_cells.items():
        i, j = cell
        reshaped_policy[i][j] = policy[index]

    # action_space = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    directions = ["-", "→", "←", "↓", "↑", "↘", "↙", "↗", "↖"]
    labels = [[" " for _ in range(len(ship.board))] for _ in range(len(ship.board))]
    for i in range(len(ship.board)):
        for j in range(len(ship.board)):
            if (i, j) in polyIter.open_cells:
                labels[i][j] = directions[int(reshaped_policy[i][j])]

    sns.heatmap(reshaped_policy, vmax = 10, annot=labels, fmt="s")
    plt.show()


def show_values(botPos):
    ship = Ship(fromFile = True)
    printBoard(ship.board)
    polyIter = PolicyIteration(ship.board)

    values = np.load(f"values.npy")
    bot_index = polyIter.open_cells[botPos]
    values = values[bot_index]
    reshaped_values = np.full((len(ship.board), len(ship.board)), np.nan)

    for cell, index in polyIter.open_cells.items():
        i, j = cell
        reshaped_values[i][j] = values[index]

    sns.heatmap(reshaped_values, annot=True, fmt=".2f")
    plt.show()


def train():
    # initialize gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 


if __name__ == "__main__":
    main()
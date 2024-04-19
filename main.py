from game.game import Game, State
from visual import Visual
import sys
import json
import datetime

from game.ship import Ship, printBoard
from game.calc.no_bot import t_no_bot, uev
from game.calc.with_bot import WithBot

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
        case "show_data":
            show_data()
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
    print(uev(expected_time))
    sns.heatmap(expected_time, vmax=300, annot=True, fmt=".2f")
    plt.show()

    
def calc_t_bot(use_existing):
    ship = Ship(fromFile = True)
    printBoard(ship.board)
        
    if use_existing:
        values = np.load(f"values.npy")
        policy = np.load(f"policy.npy")
        wb = WithBot(ship.board, values, policy)
    else:
        wb = WithBot(ship.board)
        
    values, policy = wb.policy_iteration()

    # save the values and policy matrices
    np.save(f"values.npy", values)
    np.save(f"policy.npy", policy)

    # fix one bot position, and reshape the values array, and plot it
    botPos = (0, 0)
    bot_index = wb.open_cells[botPos]
    values = values[bot_index]
    reshaped = np.full((len(ship.board), len(ship.board)), np.nan)

    for cell, index in wb.open_cells.items():
        i, j = cell
        reshaped[i][j] = values[index]

    sns.heatmap(reshaped, annot=True, fmt=".2f")
    plt.show()


def show_data():
    ship = Ship(fromFile = True)
    printBoard(ship.board)
    wb = WithBot(ship.board)

    values = np.load(f"values.npy")
    policy = np.load(f"policy.npy")

    botPos = (0, 0)
    bot_index = wb.open_cells[botPos]
    values = values[bot_index]
    reshaped_values = np.full((len(ship.board), len(ship.board)), np.nan)
    
    crewPos = (2,8)
    crew_index = wb.open_cells[crewPos]
    policy = policy[crew_index]
    reshaped_policy = np.full((len(ship.board), len(ship.board)), np.nan)

    for cell, index in wb.open_cells.items():
        i, j = cell
        reshaped_values[i][j] = values[index]
        reshaped_policy[i][j] = policy[index]

    sns.heatmap(reshaped_values, annot=True, fmt=".2f")
    plt.show()

    sns.heatmap(reshaped_policy, annot=True, fmt=".2f")
    plt.show()


if __name__ == "__main__":
    main()
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
            calc_t_bot()
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

    
def calc_t_bot():
    ship = Ship(fromFile = True)
    printBoard(ship.board)

    wb = WithBot(ship.board)
    values, policy = wb.policy_iteration()

    # save the values and policy matrices
    np.save(f"values.npy", values)
    np.save(f"policy.npy", policy)

    # fix one bot position, and reshape the values array, and plot it
    botPos = (5, 3)
    bot_index = wb.open_cells[botPos]
    values = values[bot_index]
    reshaped = np.full((len(ship.board), len(ship.board)), np.nan)
    print(len(values))

    for cell, index in wb.open_cells.items():
        i, j = cell
        reshaped[i][j] = values[index]

    sns.heatmap(reshaped, annot=True, fmt=".2f")
    plt.show()




if __name__ == "__main__":
    main()
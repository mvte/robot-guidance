import numpy as np
from game.ship import Ship
from game.calc.with_bot import WithBot


# policy contains no moves that the bot is not allowed to make
def valid_policy(policy, wb):
    for (b1, b2), (c1, c2) in wb.cell_pairs:
        b_ind = wb.open_cells[(b1, b2)]
        c_ind = wb.open_cells[(c1, c2)]
        allowed_actions = wb.compute_bot_actions((b1, b2), (c1, c2))
        if wb.action_space[int(policy[b_ind][c_ind])] not in allowed_actions:
            print("NOT ALLOWED ACTION", (b1, b2), (c1, c2), wb.action_space[policy[b_ind][c_ind]], allowed_actions)
            return False
    return True


def main():
    # load the policy
    policy = np.load("policy.npy")

    # create ship and with bot object
    ship = Ship(fromFile=True)
    wb = WithBot(ship.board)

    # check if the policy is valid
    print(valid_policy(policy, wb))


if __name__ == "__main__":
    main()
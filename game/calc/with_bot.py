import numpy as np
from game.ship import Node


action_space = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

def policy_iteration(board):
    # build a dictionary of open cells
    open_cells = {}
    index = 0
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j] == Node.OPEN:
                open_cells[(i,j)] = index
                index += 1
            elif board[i][j] == Node.TP:
                tp = (i,j)
                open_cells[(i,j)] = index
                index += 1
        
    n = len(open_cells)

    # the value of the value function matrix correspond to the expected time to reach the teleport pad
    value_function = np.zeros((n, n))
    # the value of the policy matrix corresponds to the index of the action in the action space
    policy = np.zeros((n, n))

    
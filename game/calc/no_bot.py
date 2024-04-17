import numpy as np
from game.ship import Node

def t_no_bot(board):
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

    # construct the system of equations
    A = np.zeros((n,n))
    b = np.full(n, 1)

    for cell, index in open_cells.items():
        i, j = cell
        A[index, index] = 1

        if cell == tp:
            b[index] = 0
            continue

        b[index] = 1
        neighbors = [(i+1,j), (i-1,j), (i,j+1), (i,j-1)]
        open_neighbors = [n for n in neighbors if n in open_cells]

        for neighbor in open_neighbors:
            A[index, open_cells[neighbor]] = -1/len(open_neighbors)

    # solve the system of equations
    x = np.linalg.solve(A, b)

    # build the expected time matrix
    expected_time = np.full((len(board), len(board)), np.nan)
    for cell, index in open_cells.items():
        i, j = cell
        expected_time[i][j] = x[index]

    return expected_time


# uniform expected value for each cell in a given ndarray
def uev(nda):
    sum = np.nansum(nda)
    # - 1 because we don't start on the teleport pad
    count = nda.size - np.sum(np.isnan(nda)) - 1 

    return sum / count



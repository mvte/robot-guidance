import random
import json
from enum import Enum

'''
a ship is a boolean grid of size dim x dim
if a node is True, then that node is open and crewmates/aliens/bots can move into it
if a node is False, then that node is closed and these^ cannot move into it
'''
class Ship:

    def __init__(self, dim=11, fromFile=False):
        # print("creating ship")

        if not fromFile:
            tempBoard = generate_ship(dim)
        else:
            with open("ship.json", "r") as f:
                data = json.load(f)
                dim = data["dim"]
                closed = data["closed"]
            tempBoard = generate_ship_from_file(dim, closed)
        
        self.board = tempBoard
        self.dim = dim

    def getShip(self):
        return self.board
    

    def getValidMoves(self, pos):
        x, y = pos
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        validMoves = []
        for dir in dirs:
            dx, dy = dir
            if x + dx < 0 or x + dx >= len(self.board) or y + dy < 0 or y + dy >= len(self.board):
                continue
            if self.board[x + dx][y + dy] != Node.CLOSED:
                validMoves.append((x + dx, y + dy))
        return validMoves
    

class Node(Enum):
    OPEN = 1
    CLOSED = 2
    TP = 3


def generate_ship(dim):
    ship = [[Node.OPEN for i in range(dim)] for j in range(dim)]

    # teleport pad at center of the ship
    mdpt = dim // 2
    ship[mdpt][mdpt] = Node.TP
    
    # 4 closed nodes at corners around the teleport pad
    diags = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
    for diag in diags:
        dx, dy = diag
        ship[mdpt + dx][mdpt + dy] = Node.CLOSED

    # 10 random closed nodes
    for i in range(10):
        x = random.randint(0, dim - 1)
        y = random.randint(0, dim - 1)
        while ship[x][y] == Node.CLOSED or (abs(x - mdpt) < 3 and abs(y - mdpt) < 3):
            x = random.randint(0, dim - 1)
            y = random.randint(0, dim - 1)

        ship[x][y] = Node.CLOSED
    
    # make sure that the teleport pad entrances are not obstructed
    return ship


def generate_ship_from_file(dim, closed):
    ship = [[Node.OPEN for i in range(dim)] for j in range(dim)]
    for cell in closed:
        x, y = cell
        ship[x][y] = Node.CLOSED
    
    mdpt = dim // 2
    ship[mdpt][mdpt] = Node.TP
    
    return ship



def printBoard(ship):
    for row in ship:
        for node in row:
            print("[", end="")
            if node == Node.OPEN:
                print(" ", end="")
            elif node == Node.CLOSED:
                print("#", end="")
            else:
                print("T", end="")
            print("]", end="")
        print()

def generate_ships_parallel(dim, num_ships):
    pass
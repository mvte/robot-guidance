import random

class Crewmate:

    def __init__(self, pos):
        self.pos = pos


    def computeNextStep(self, ship, botPos=None):
        # if the bot is adjacent to the crewmate, move as far away as possible
        x, y = self.pos
        moves = ship.getValidMoves(self.pos)

        if not botPos:
            return random.choice(moves)

        botX, botY = botPos
        if botPos and abs(botX - x) <= 1 and abs(botY - y) <= 1:
            maxDist = 0
            bestMove = None
            for move in moves:
                dx, dy = moves
                dist = abs(botX - dx) + abs(botY - dy)
                if dist > maxDist:
                    maxDist = dist
                    bestMove = move
            return bestMove

        # if the bot is not adjacent to the crewmate, pick a random direction
        return random.choice(moves)

    

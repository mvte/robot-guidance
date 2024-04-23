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
        if botPos and abs(x - botX) + abs(y - botY) == 1:
            maxDist = 0
            bestMoves = []
            for move in moves:
                nx, ny = move
                dist = abs(botX - nx) + abs(botY - ny)
                if dist == maxDist:
                    bestMoves.append(move)
                if dist > maxDist:
                    maxDist = dist
                    bestMoves = [move]
            return random.choice(bestMoves)

        # if the bot is not adjacent to the crewmate, pick a random direction
        return random.choice(moves)

    

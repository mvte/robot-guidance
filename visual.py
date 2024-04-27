import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from game.ship import Node
from game.game import State


'''
displays the game state
'''
class Visual:

    def __init__(self, game):
        self.fig, self.ax = plt.subplots()
        self.game = game


    def update(self):
        if self.game.state != State.RUNNING:
            return
        
        self.ax.clear()

        remapped = self._remap(self.game.sims[0].ship.board)

        sim = sns.heatmap(
            remapped, 
            ax=self.ax, 
            vmax=5,
            cbar=False, 
            square=True, 
            xticklabels=False,
            yticklabels=False,
        )

        self.fig.canvas.draw()

        plt.pause(0.01)
    

    def _remap(self, board):
        lookup = {
            Node.CLOSED: 1,
            Node.OPEN: np.nan,
            Node.TP: 5
        }

        arr = np.array([[lookup[node] for node in row] for row in board])
        
        if self.game.sims[0].bot:
            botPos = self.game.sims[0].bot.pos
            arr[botPos[0]][botPos[1]] = 3

        crewPos = self.game.sims[0].crew.pos
        arr[crewPos[0]][crewPos[1]] = 4

        return arr


if __name__ == "__main__":
    print("testing")

    

# controlled agents are controlled by the player, or a neural network in training
class ControlledBot():
    def __init__(self, pos):
        self.pos = pos

    def computeNextStep(self, ship, crewPos):
        return self.pos
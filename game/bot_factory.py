from game.agents.optimal import OptimalBot
from game.agents.learned import LearnedBot

def botFactory(which, pos, ship):
    match which:
        case "optimal":
            return OptimalBot(pos, ship)
        case "learned":
            return LearnedBot(pos, ship)
        case _:
            return None
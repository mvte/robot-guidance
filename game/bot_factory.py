from game.agents.optimal import OptimalBot
from game.agents.learned import LearnedBot 
from game.agents.controlled import ControlledBot
from game.agents.generalized import GeneralizedBot

def botFactory(which, pos, ship):
    match which:
        case "optimal":
            return OptimalBot(pos, ship)
        case "learned":
            return LearnedBot(pos, ship)
        case "controlled":
            return ControlledBot(pos)
        case "generalized":
            return GeneralizedBot(pos, ship)
        case _:
            return None
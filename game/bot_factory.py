from game.agents.optimal import OptimalBot

def botFactory(which, pos, ship):
    match which:
        case "optimal":
            return OptimalBot(pos, ship)
        case _:
            return None
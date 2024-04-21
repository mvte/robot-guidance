import numpy as np
from game.ship import Node

# policy iteration
class WithBot:
    action_space = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    def __init__(self, board, values=None, policy=None):
        self.board = board

        cell_pairs = {}
        index = 0
        for b1 in range(len(board)):
            for b2 in range(len(board)):
                if board[b1][b2] == Node.CLOSED:
                    continue
                for c1 in range(len(board)):
                    for c2 in range(len(board)):
                        if board[c1][c2] == Node.CLOSED:
                            continue
                        if (b1, b2) == (c1, c2):
                            continue
                        cell_pairs[((b1, b2), (c1, c2))] = index
                        index += 1
        
        self.num_pairs = len(cell_pairs)
        self.cell_pairs = cell_pairs
        self.bot_actions = {}
        self.crew_responses = {}

        open_cells = {}
        index = 0
        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] == Node.OPEN:
                    open_cells[(i,j)] = index
                    index += 1
                elif board[i][j] == Node.TP:
                    self.tp = (i,j)
                    open_cells[(i,j)] = index
                    index += 1

        self.n = len(open_cells)
        self.open_cells = open_cells

        # each entry of the values matrix correspond to the expected time to reach the teleport pad
        self.values = np.zeros((self.n, self.n)) if values is None else values
        # each entry of the policy matrix corresponds to the index of the action in the action space
        # this is the initial policy
        self.policy = np.zeros((self.n, self.n)) if policy is None else policy


    def policy_evaluation(self, policy):
        A = np.zeros((self.num_pairs, self.num_pairs))
        b = np.full(self.num_pairs, 1)

        for ((b1, b2), (c1, c2)), ind in self.cell_pairs.items():
            A[ind][ind] = 1
            if (c1, c2) == self.tp:
                b[ind] = 0
                continue

            b_ind = self.open_cells[(b1, b2)]
            c_ind = self.open_cells[(c1, c2)]

            # determine the action 
            action = self.action_space[int(policy[b_ind][c_ind])]
            # determine the next state (the action should ALWAYS be valid)
            b_act = (b1 + action[0], b2 + action[1])
            # determine the set of all crew responses 
            s_r = self.compute_crew_responses(b_act, (c1, c2))
            p = 1 / len(s_r)

            # update the system of equations
            for s in s_r:
                s_ind = self.cell_pairs[b_act, s]
                A[ind, s_ind] = -p
        
        x = np.linalg.solve(A, b)
        
        value = np.zeros((self.n, self.n))
        for (b, c), ind in self.cell_pairs.items():
            value[self.open_cells[b]][self.open_cells[c]] = x[ind]
                
        return value


    def policy_iteration(self):
        stable = False
        num_iter = 0
        while not stable:
            stable = True
            # check for convergence of the values matrix 
            values_prev = self.values.copy()
            values = self.policy_evaluation(self.policy)
            self.values = values
            if num_iter != 0 and np.allclose(values, values_prev, rtol=1e-10, atol=1e-12):
                print("values arbitrarily close: stopping policy iteration")
                break
            
            # iterate through all possible states
            for (b1, b2), (c1, c2) in self.cell_pairs:
                b_ind = self.open_cells[(b1, b2)]
                c_ind = self.open_cells[(c1, c2)]

                # determine the set of all possible actions
                allowed_actions = self.compute_bot_actions((b1, b2), (c1, c2))

                # compute the action values for each action
                action_values = [
                    self.compute_action_value((b1, b2), (c1, c2), values, action, allowed_actions) 
                    for action in self.action_space
                ]

                # determine the action that minimizes the action values
                action_values_np = np.array(action_values)
                new_action = int(np.argmin(action_values_np))

                # check if the action has changed
                if new_action != self.policy[b_ind][c_ind]:
                    stable = False

                # update the action in the policy matrix
                self.policy[b_ind][c_ind] = new_action

            num_iter += 1

            # save the values and policy matrices every 50 iterations
            if num_iter % 50 == 0:
                print("saving values and policy matrices")
                np.save(f"data/values-ch{num_iter//5}.npy", values)
                np.save(f"data/policy-ch{num_iter//5}.npy", self.policy)

            print("policy iteration", num_iter)
    
        return values, self.policy

    def is_adj(self, a, b):
        # only in cardinal directions
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
    

    def compute_bot_actions(self, botPos, crewPos):
        # memoization
        if (botPos, crewPos) in self.bot_actions:
            return self.bot_actions[(botPos, crewPos)]
    
        actions = []
        for action in self.action_space:
            new_pos = (botPos[0] + action[0], botPos[1] + action[1])
            if new_pos in self.open_cells and new_pos != crewPos:
                actions.append(action)
        
        self.bot_actions[(botPos, crewPos)] = actions
        return actions
    

    def compute_action_value(self, botPos, crewPos, values, action, allowed_actions):
        if action not in allowed_actions:
            return np.inf

        b_action = (botPos[0] + action[0], botPos[1] + action[1])
        s_r = self.compute_crew_responses(b_action, crewPos)

        return float(1 + sum([
            values[self.open_cells[b_action]][self.open_cells[s]] / len(s_r) for s in s_r
        ]))
        


    # the crewmate randomly chooses the cell that maximizes the distance from the bot if they're adjacent
    def compute_crew_responses(self, botPos, crewPos):
        # memoization
        if (botPos, crewPos) in self.crew_responses:
            return self.crew_responses[(botPos, crewPos)]

        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        adj_cells = []
        
        for dir in dirs:
            dx, dy = dir
            if (crewPos[0] + dx, crewPos[1] + dy) in self.open_cells:
                adj_cells.append((crewPos[0] + dx, crewPos[1] + dy))
        
        if self.is_adj(crewPos, botPos):
            max_cells = []
            max_dist = 0
            for cell in adj_cells:
                dist = self._getManhattanDistance(cell, botPos)
                if dist > max_dist:
                    max_dist = dist
                    max_cells = [cell]
                elif dist == max_dist:
                    max_cells.append(cell)
                    
            responses = max_cells
        else:
            responses = adj_cells
    
        self.crew_responses[(botPos, crewPos)] = responses
        return responses

    def _getManhattanDistance(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        return abs(x1 - x2) + abs(y1 - y2)

            

    




    
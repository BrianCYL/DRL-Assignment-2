import math
import random
from collections import defaultdict
import copy
import numpy as np

class NTupleApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want
        """
        self.board_size = board_size
        self.patterns = patterns
        # Create a weight dictionary for each pattern (shared within a pattern group)
        self.weights = [defaultdict(float) for _ in patterns]
        
        # Generate symmetrical transformations for each pattern
        # self.symmetry_patterns = patterns
        self.symmetry_patterns = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            for sym_ in syms:
                self.symmetry_patterns.append(sym_)

    def generate_symmetries(self, pattern):
        # TODO: Generate 8 symmetrical transformations of the given pattern.
        rot90_pattern = [(c, 3 - r) for (r, c) in pattern] # rot90
        rot180_pattern = [(3 - r, 3 - c) for (r, c) in pattern] # rot180
        rot270_pattern = [(3 - c, r) for (r, c) in pattern] # rot270
        flip_pattern = [(r, 3 - c) for (r, c) in pattern] # flip
        flip_rot90_pattern = [(c, 3 - r) for (r, c) in flip_pattern] # flip + rot90
        flip_rot180_pattern = [(3 - r, 3 - c) for (r, c) in flip_pattern] # flip + rot180
        flip_rot270_pattern = [(3 - c, r) for (r, c) in flip_pattern] # flip + rot270
        return [pattern, rot90_pattern, rot180_pattern, rot270_pattern, flip_pattern, flip_rot90_pattern, flip_rot180_pattern, flip_rot270_pattern]

    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
        # TODO: Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
        tile_values = [board[r, c] for (r, c) in coords]
        return tuple(tile_values)

    def value(self, board):
        # TODO: Estimate the board value: sum the evaluations from all patterns.
        pattern_val = 0.0
        for i, patterns in enumerate(self.symmetry_patterns):
            tile_to_index = tuple(self.tile_to_index(tile) for tile in self.get_feature(board, patterns))
            pattern_val += self.weights[i//8][tile_to_index]
            # pattern_val += self.weights[i][tile_to_index]
        return pattern_val
    
    def update(self, board, delta, alpha):
        # TODO: Update weights based on the TD error.
        update_val = alpha * delta / len(self.symmetry_patterns)
        for i, pattern in enumerate(self.symmetry_patterns):
            tile_to_index = tuple(self.tile_to_index(tile) for tile in self.get_feature(board, pattern))
            self.weights[i//8][tile_to_index] += update_val
            # self.weights[i][tile_to_index] += update_val

# # # Note: This MCTS implementation is almost identical to the previous one,
# # # except for the rollout phase, which now incorporates the approximator.

# # # Node for TD-MCTS using the TD-trained value approximator
# class TD_MCTS_Node:
#     def __init__(self, env, state, score, parent=None, action=None):
#         """
#         state: current board state (numpy array)
#         score: cumulative score at this node
#         parent: parent node (None for root)
#         action: action taken from parent to reach this node
#         """
#         self.state = state
#         self.score = score
#         self.parent = parent
#         self.action = action
#         self.children = {}
#         self.visits = 0
#         self.total_reward = score
#         # List of untried actions based on the current state's legal moves
#         self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]

#     def fully_expanded(self):
#         # A node is fully expanded if no legal actions remain untried.
#         return len(self.untried_actions) == 0



# # TD-MCTS class utilizing a trained approximator for leaf evaluation
# class TD_MCTS:
#     def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=0.99):
#         self.env = env
#         self.approximator = approximator
#         self.iterations = iterations
#         self.c = exploration_constant
#         self.rollout_depth = rollout_depth
#         self.gamma = gamma

#     def create_env_from_state(self, state, score):
#         # Create a deep copy of the environment with the given state and score.
#         new_env = copy.deepcopy(self.env)
#         new_env.board = state.copy()
#         new_env.score = score
#         return new_env

#     def select_child(self, node):
#         # TODO: Use the UCT formula: Q + c * sqrt(log(parent.visits)/child.visits) to select the best child.
#         uct_value = -float('inf')
#         best_child = None
#         for child in node.children.values():
#             if child.visits == 0:
#                 return child
#             uct = (child.total_reward / child.visits) + self.c * np.sqrt(np.log(node.visits) / child.visits)
#             if uct > uct_value:
#                 uct_value = uct
#                 best_child = child
#         # print(f"Best child: {best_child.action}, visits: {best_child.visits}, uct: {uct_value}")
#         return best_child

#     def rollout(self, sim_env, depth):
#         # TODO: Perform a random rollout until reaching the maximum depth or a terminal state.
#         # TODO: Use the approximator to evaluate the final state.
#         steps = 0
#         done = False
#         prev_score = sim_env.score
#         reward = 0
#         while steps < depth and not done:
#             legal_moves = [action for action in range(sim_env.action_space.n) if sim_env.is_move_legal(action)]
#             if not legal_moves:
#                 break
#             action = np.random.choice(legal_moves)
#             _, score, done, _, _ = sim_env.step(action)
#             if done:
#                 return reward / 20000
#             reward += (self.gamma ** steps) * (score - prev_score)
#             steps += 1
#             prev_score = score
#         return (reward + (self.gamma ** steps) * self.approximator.value(sim_env.after_state)) / 20000
        
        
#     def backpropagate(self, node, reward):
#         # TODO: Propagate the obtained reward back up the tree.
#         while node is not None:
#             node.visits += 1
#             node.total_reward += (reward - node.total_reward) / node.visits
#             node = node.parent
#             # reward *= self.gamma

#     def run_simulation(self, root):
#         node = root
#         sim_env = self.create_env_from_state(node.state, node.score)
#         prev_score = sim_env.score
#         # TODO: Selection: Traverse the tree until reaching a non-fully expanded node.
#         while node.fully_expanded() and node.children:
#             node = self.select_child(node)
#             _,_,_,_,_ = sim_env.step(node.action)

#         # TODO: Expansion: if the node has untried actions, expand one.
#         action = np.random.choice(node.untried_actions)
#         state, score, done, _, after_state = sim_env.step(action)

#         child = TD_MCTS_Node(self.env, copy.deepcopy(after_state), self.approximator.value(after_state), parent=node, action=action)
#         node.untried_actions.remove(action)
#         node.children[action] = child
#         node = child
#         sim_env = self.create_env_from_state(node.state, node.score)

#         # Rollout: Simulate a random game from the expanded node.
#         rollout_reward = self.rollout(sim_env, self.rollout_depth)
#         # Backpropagation: Update the tree with the rollout reward.
#         # print(reward, rollout_reward)
#         self.backpropagate(node, rollout_reward)

#     def best_action_distribution(self, root):
#         # Compute the normalized visit count distribution for each child of the root.
#         total_visits = sum(child.visits for child in root.children.values())
#         distribution = np.zeros(4)
#         best_visits = -1
#         best_action = None
#         for action, child in root.children.items():
#             distribution[action] = child.visits / total_visits if total_visits > 0 else 0
#             if child.visits > best_visits:
#                 best_visits = child.visits
#                 best_action = action
#         return best_action, distribution

# class DecisionNode:
#     def __init__(self, env, state, parent=None):
#         self.env = copy.deepcopy(env)
#         self.state = state
#         self.parent = parent
#         self.children = {}
#         self.visits = 0
#         self.total_reward = 0.0
#         self.untried_actions = [a for a in range(4) if self.env.is_move_legal(a)]
#         self.action = None
    
#     def fully_expanded(self):
#         # A node is fully expanded if no legal actions remain untried.
#         return len(self.untried_actions) == 0

# class MCTSNode:
#     def __init__(self, parent=None):
#         self.parent = parent
#         self.children = {}
#         self.visits = 0
#         self.total_reward = 0.0

#     def update(self, reward):
#         self.visits += 1
#         self.total_reward += (reward - self.total_reward) / self.visits


# class DecisionNode(MCTSNode):
#     def __init__(self, env, state, score, parent=None, action=None):
#         super().__init__(parent)
#         self.state = state
#         self.score = score
#         self.action = action
#         self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]

#     def fully_expanded(self):
#         return len(self.untried_actions) == 0


# class ChanceNode(MCTSNode):
#     def __init__(self, env, state, score, parent=None):
#         super().__init__(parent)
#         self.state = state
#         self.score = score
#         self.spawn_outcomes = self.get_possible_spawns(env)

#     def get_possible_spawns(self, env):
#         empty = list(zip(*np.where(env.board == 0)))
#         outcomes = []
#         for pos in empty:
#             for val, prob in [(2, 0.9), (4, 0.1)]:
#                 outcomes.append(((pos, val), prob / len(empty)))
#         return outcomes
    
# class TD_MCTS:
#     def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=0.99):
#         self.env = env
#         self.approximator = approximator
#         self.iterations = iterations
#         self.c = exploration_constant
#         self.rollout_depth = rollout_depth
#         self.gamma = gamma

#     def create_env_from_state(self, state, score):
#         env_copy = copy.deepcopy(self.env)
#         env_copy.board = state.copy()
#         env_copy.score = score
#         return env_copy

#     def uct_score(self, parent, child):
#         if child.visits == 0:
#             return float('inf')
#         exploitation = child.total_reward / child.visits
#         exploration = self.c * np.sqrt(np.log(parent.visits) / child.visits)
#         return exploitation + exploration

#     def select(self, node, sim_env):
#         while isinstance(node, DecisionNode) and node.fully_expanded() and node.children:
#             node = max(node.children.values(), key=lambda child: self.uct_score(node, child))
#             # _, _, _, _, _ = sim_env.step(node.action)

#             if isinstance(node, ChanceNode) and node.children:
#                 # Sample one possible tile
#                 (pos, val), _ = random.choice(node.spawn_outcomes)
#                 sim_env.board[pos] = val
#                 node = node.children.get((pos, val), None)
#                 if node is None:
#                     break
#         return node

#     def expand(self, node, sim_env):
#         if isinstance(node, DecisionNode) and node.untried_actions:
#             action = random.choice(node.untried_actions)
            
#             _, score, done, _, after_state = sim_env.step(action)

#             child = ChanceNode(self.env, copy.deepcopy(after_state), score, parent=node)
#             node.untried_actions.remove(action)
#             node.children[action] = child
#             return child

#         elif isinstance(node, ChanceNode):
#             if not node.spawn_outcomes:
#                 return node
#             (pos, val), prob = random.choices(
#                 node.spawn_outcomes,
#                 weights=[p for (_, p) in node.spawn_outcomes]
#             )[0]
#             sim_env.board[pos] = val
#             state = sim_env.board.copy()
#             child = DecisionNode(self.env, copy.deepcopy(state), sim_env.score, parent=node)
#             node.children[(pos, val)] = child
#             return child

#         return node  # already a leaf

#     def rollout(self, sim_env, depth):
#         steps = 0
#         done = False
#         prev_score = sim_env.score
#         total_reward = 0

#         while steps < depth and not done:
#             legal = [a for a in range(sim_env.action_space.n) if sim_env.is_move_legal(a)]
#             if not legal:
#                 break
#             action = random.choice(legal)
#             _, score, done, _, _ = sim_env.step(action)
#             reward = score - prev_score
#             total_reward += (self.gamma ** steps) * reward
#             prev_score = score
#             steps += 1

#         approx_value = self.approximator.value(sim_env.after_state)
#         return (total_reward + (self.gamma ** steps) * approx_value) / 20000

#     def backpropagate(self, node, reward):
#         while node is not None:
#             node.update(reward)
#             node = node.parent

#     def run_simulation(self, node):
#         sim_env = self.create_env_from_state(copy.deepcopy(node).state, node.score)
#         node = self.select(node, sim_env)
#         node = self.expand(node, sim_env)
#         reward = self.rollout(sim_env, self.rollout_depth)
#         self.backpropagate(node, reward)
    
#     def best_action_distribution(self, root):
#         total_visits = sum(child.visits for child in root.children.values())
#         distribution = np.zeros(4)
#         best_visits = -1
#         best_action = None
#         for action, child in root.children.items():
#             distribution[action] = child.visits / total_visits if total_visits > 0 else 0
#             if child.visits > best_visits:
#                 best_visits = child.visits
#                 best_action = action
#         return best_action, distribution

# import numpy as np
# import copy
# import math
# import random
# from collections import defaultdict

class DecisionNode:
    """
    A decision (action) node. This represents the state that the agent sees (after a random tile has been added).
    """
    def __init__(self, env, parent=None, action_from_parent=None):
        # Make a deep copy so that each node's environment is independent.
        self.env = copy.deepcopy(env)
        self.parent = parent
        self.action_from_parent = action_from_parent
        # Children: mapping from action (0,1,2,3) to a ChanceNode.
        self.children = {}
        self.visits = 0
        self.value = 0.0
        # Untried actions (only legal moves)
        self.untried_actions = [action for action in range(4) if self.env.is_move_legal(action)]
        # Terminal flag (if game is over)
        self.is_terminal = self.env.is_game_over()

    def uct_select_child(self, exploration=math.sqrt(2)):
        """Select a child chance node using the UCT formula."""
        best_value = -float('inf')
        best_child = None
        for action, child in self.children.items():
            if child.visits == 0:
                uct = float('inf')
            else:
                uct = child.value / child.visits + exploration * math.sqrt(math.log(self.visits) / child.visits)
            if uct > best_value:
                best_value = uct
                best_child = child
        return best_child


class ChanceNode:
    """
    A chance node that represents the intermediate state after the player's move (before the random tile is added).
    Its children are decision nodes corresponding to each possible outcome of the tile addition.
    """
    def __init__(self, env, parent, action, reward=0):
        self.env = copy.deepcopy(env)  # This state results from taking an action (without spawning a tile)
        self.parent = parent
        self.action = action
        self.reward = reward  # Store the reward obtained from the parent to this node
        # Children: mapping from outcome to decision node.
        # Each outcome is a tuple: (x, y, tile, probability)
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.untried_outcomes = self._get_possible_outcomes()
        # Check terminality: if the board is full, no tile can be added.
        self.is_terminal = self.env.is_game_over() or (len(self.untried_outcomes) == 0)

    def _get_possible_outcomes(self):
        outcomes = []
        board = self.env.board
        empty_cells = list(zip(*np.where(board == 0)))
        if empty_cells:
            prob2 = 0.9 / len(empty_cells)
            prob4 = 0.1 / len(empty_cells)
            for cell in empty_cells:
                outcomes.append((cell[0], cell[1], 2, prob2))
                outcomes.append((cell[0], cell[1], 4, prob4))
        return outcomes

    def sample_outcome(self):
        """
        When the chance node is fully expanded, sample one of its decision-node children
        according to the tile addition probabilities.
        """
        outcomes = []
        weights = []
        for outcome, child in self.children.items():
            outcomes.append(child)
            weights.append(outcome[3])  # outcome[3] holds the probability weight
        # In case all outcomes have zero weight (should not happen), choose uniformly.
        if sum(weights) == 0:
            return random.choice(outcomes)
        chosen = random.choices(outcomes, weights=weights, k=1)[0]
        return chosen


class MCTS:
    """
    Modified MCTS for the 2048 game using afterstate value function.
    Alternates between decision nodes (agent moves) and chance nodes (random tile additions).
    """
    def __init__(self, env, approximator, iterations=1000, exploration=0, value_norm=20000):
        self.root = DecisionNode(env)
        self.approximator = approximator
        self.iterations = iterations
        self.exploration = exploration
        self.value_norm = value_norm  # Normalization constant for the value function

    def search(self):
        """
        Run MCTS for a given number of iterations and return the best action from the root.
        """
        # First, fully expand the root node to compute all possible afterstates
        if not self.root.children:
            self._expand_decision_node(self.root)
            
        for _ in range(self.iterations):
            # Selection: traverse the tree to a leaf node
            leaf, path, cumulative_reward = self._tree_policy(self.root)
            # Evaluation: use the approximator instead of rollout
            value = self._evaluate_node(leaf, cumulative_reward)
            # Backpropagation: update the nodes along the path with the evaluation
            self._backpropagate(path, value)
            
        # Select the action leading to the child chance node with the highest visit count
        best_action = None
        best_visits = -1
        for action, chance_node in self.root.children.items():
            if chance_node.visits > best_visits:
                best_visits = chance_node.visits
                best_action = action
        return best_action

    def _expand_decision_node(self, node):
        """
        Expand a decision node by creating all possible chance node children.
        """
        for action in node.untried_actions:
            new_env = copy.deepcopy(node.env)
            # Perform the move WITHOUT spawning a random tile
            reward = new_env.step(action, spawn_tile=False)[1]  # Get the reward from the action
            # Create a chance node corresponding to the deterministic outcome
            chance_child = ChanceNode(new_env, node, action, reward)
            node.children[action] = chance_child
            # Calculate the value of this afterstate using the approximator
            chance_child.value = self.approximator.value(chance_child.env.board)
        # Remove all untried actions since we've expanded all of them
        node.untried_actions = []

    def _expand_chance_node(self, node):
        """
        Expand a chance node by creating all possible decision node children.
        """
        for outcome in node.untried_outcomes:
            new_env = copy.deepcopy(node.env)
            x, y, tile, _ = outcome
            new_env.board[x, y] = tile  # simulate the tile addition
            decision_child = DecisionNode(new_env, parent=node, action_from_parent=outcome)
            node.children[outcome] = decision_child
        # Remove all untried outcomes since we've expanded all of them
        node.untried_outcomes = []

    def _tree_policy(self, node):
        """
        Traverse the tree (starting at a decision node) and return the leaf node,
        the path from the root to the leaf, and the cumulative reward along that path.
        """
        path = [node]
        current = node
        cumulative_reward = 0
        
        while not current.is_terminal:
            # If we are at a decision node:
            if isinstance(current, DecisionNode):
                # If there are untried actions, expand all of them at once
                if current.untried_actions:
                    self._expand_decision_node(current)
                
                # Select a chance node using UCT
                chance_child = current.uct_select_child(self.exploration)
                cumulative_reward += chance_child.reward  # Add reward from this action
                path.append(chance_child)
                current = chance_child
                
            # If we are at a chance node:
            elif isinstance(current, ChanceNode):
                # If there are untried outcomes, expand all of them at once
                if current.untried_outcomes:
                    self._expand_chance_node(current)
                    
                # Sample a decision node child according to the outcome probabilities
                decision_child = current.sample_outcome()
                path.append(decision_child)
                current = decision_child
                
            # Stop if we reach a leaf node (terminal or newly expanded)
            if (isinstance(current, DecisionNode) and 
                (current.is_terminal or not current.children)) or \
               (isinstance(current, ChanceNode) and not current.children):
                break
                
        return current, path, cumulative_reward

    def _evaluate_node(self, node, cumulative_reward):
        """
        Evaluate a node using the value function instead of a rollout.
        """
        if isinstance(node, DecisionNode):
            # For a decision node, we need to expand its afterstates first if not already done
            if not node.children and not node.is_terminal:
                self._expand_decision_node(node)
                
            # If terminal or no legal moves, the value is 0
            if node.is_terminal or not node.children:
                node_value = 0
            else:
                # Value is the maximum of (reward + afterstate value) across all actions
                max_value = float('-inf')
                for action, chance_node in node.children.items():
                    action_value = chance_node.reward + chance_node.value
                    max_value = max(max_value, action_value)
                node_value = max_value
                
        elif isinstance(node, ChanceNode):
            # For a chance node, we can use its already computed value
            node_value = node.value
            
        # Return the normalized value: (cumulative_reward + node_value) / value_norm
        return (cumulative_reward + node_value) / self.value_norm

    def _backpropagate(self, path, value):
        """
        Update the statistics of all nodes along the path with the evaluation value.
        """
        for node in reversed(path):
            node.visits += 1
            node.value += value  # Update node's total value
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
        self.pattern_idx = []
        for idx, pattern in enumerate(self.patterns):
            syms = self.generate_symmetries(pattern)
            for sym_ in syms:
                self.symmetry_patterns.append(sym_)
                self.pattern_idx.append(idx)

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
        for idx, patterns in zip(self.pattern_idx, self.symmetry_patterns):
            tile_to_index = tuple(self.tile_to_index(tile) for tile in self.get_feature(board, patterns))
            pattern_val += self.weights[idx][tile_to_index]
            # pattern_val += self.weights[i][tile_to_index]
        return pattern_val
    
    def update(self, board, delta, alpha):
        # TODO: Update weights based on the TD error.
        update_val = alpha * delta
        for idx, patterns in zip(self.pattern_idx, self.symmetry_patterns):
            tile_to_index = tuple(self.tile_to_index(tile) for tile in self.get_feature(board, patterns))
            self.weights[idx][tile_to_index] += update_val
            # self.weights[i][tile_to_index] += update_val

# class DecisionNode:
#     def __init__(self, env, parent=None, action=None):
#         self.env = copy.deepcopy(env)
#         self.parent = parent
#         self.action = action
#         self.children = {}  # action -> ChanceNode
#         self.untried_actions = [a for a in range(4) if self.env.is_move_legal(a)]
#         self.visits = 0
#         self.value = 0.0
#         self.is_terminal = self.env.is_game_over()

#     def fully_expanded(self):
#         return len(self.untried_actions) == 0

#     def uct_select_child(self, exploration=math.sqrt(2)):
#         best_value = -float("inf")
#         best_child = None
#         for action, child in self.children.items():
#             if child.visits == 0:
#                 uct = float("inf")
#             else:
#                 uct = child.value / child.visits + exploration * math.sqrt(math.log(self.visits) / child.visits)
#             if uct > best_value:
#                 best_value = uct
#                 best_child = child
#         return best_child


# class ChanceNode:
#     def __init__(self, env, parent, action, reward):
#         self.env = copy.deepcopy(env)
#         self.parent = parent
#         self.action = action
#         self.reward = reward
#         self.children = {}  # (x, y, tile, prob) -> DecisionNode
#         self.visits = 0
#         self.value = 0.0
#         self.untried_outcomes = self._get_possible_outcomes()
#         self.is_terminal = self.env.is_game_over() or len(self.untried_outcomes) == 0

#     def _get_possible_outcomes(self):
#         empty = list(zip(*np.where(self.env.board == 0)))
#         outcomes = []
#         if empty:
#             prob2 = 0.9 / len(empty)
#             prob4 = 0.1 / len(empty)
#             for (x, y) in empty:
#                 outcomes.append((x, y, 2, prob2))
#                 outcomes.append((x, y, 4, prob4))
#         return outcomes

#     def sample_outcome(self):
#         outcomes, weights = [], []
#         for outcome, child in self.children.items():
#             outcomes.append(child)
#             weights.append(outcome[3])
#         return random.choices(outcomes, weights=weights)[0] if weights else random.choice(outcomes)

# class TD_MCTS:
#     def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, gamma=0.99):
#         self.env = env
#         self.approximator = approximator
#         self.iterations = iterations
#         self.c = exploration_constant
#         self.gamma = gamma

#     def create_env_from_state(self, state, score):
#         new_env = copy.deepcopy(self.env)
#         new_env.board = state.copy()
#         new_env.score = score
#         return new_env

#     def expand_decision_node(self, node):
#         for action in node.untried_actions:
#             env = copy.deepcopy(node.env)
#             _, reward, _, _ = env.step(action, spawn_tile=False)
#             child = ChanceNode(env, node, action, reward)
#             child.value = self.approximator.value(env.board)
#             node.children[action] = child
#         node.untried_actions = []

#     def expand_chance_node(self, node):
#         for outcome in node.untried_outcomes:
#             x, y, val, _ = outcome
#             env = copy.deepcopy(node.env)
#             env.board[x, y] = val
#             child = DecisionNode(env, parent=node, action=outcome)
#             node.children[outcome] = child
#         node.untried_outcomes = []

#     def backpropagate(self, path, value):
#         for node in reversed(path):
#             node.visits += 1
#             node.value += value

#     def run_simulation(self, root):
#         node = root
#         path = [node]
#         cumulative_reward = 0

#         while not node.is_terminal:
#             if isinstance(node, DecisionNode):
#                 if node.untried_actions:
#                     self.expand_decision_node(node)
#                 node = node.uct_select_child(self.c)
#                 path.append(node)
#                 cumulative_reward += node.reward
#             elif isinstance(node, ChanceNode):
#                 if node.untried_outcomes:
#                     self.expand_chance_node(node)
#                 node = node.sample_outcome()
#                 path.append(node)

#             # Stop at leaf
#             if (isinstance(node, DecisionNode) and (node.is_terminal or not node.children)) or (isinstance(node, ChanceNode) and not node.children):
#                 break

#         value = self.evaluate(node, cumulative_reward)
#         self.backpropagate(path, value)

#     def evaluate(self, node, cumulative_reward):
#         if isinstance(node, DecisionNode):
#             if not node.children and not node.is_terminal:
#                 self.expand_decision_node(node)
#             if node.is_terminal or not node.children:
#                 node_value = 0
#             else:
#                 node_value = max(c.reward + c.value for c in node.children.values())
#         elif isinstance(node, ChanceNode):
#             node_value = node.value
#         return (cumulative_reward + node_value) / 20000  # your original normalization

#     def best_action_distribution(self, root):
#         dist = np.zeros(4)
#         total = sum(c.visits for c in root.children.values())
#         best_action = None
#         best_visits = -1
#         for action, child in root.children.items():
#             dist[action] = child.visits / total if total > 0 else 0
#             if child.visits > best_visits:
#                 best_visits = child.visits
#                 best_action = action
#         return best_action, dist

# class DecisionNode:
#     def __init__(self, env, state, score, parent=None, action=None):
#         self.env = env
#         self.state = state     
#         self.score = score     
#         self.parent = parent   
#         self.action = action 
#         self.children = {}
#         self.visits = 0
#         self.total_reward = 0.0
#         self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]

#     def fully_expanded(self):
#         return len(self.untried_actions) == 0


# class ChanceNode:
#     def __init__(self, env, state, score, parent=None):
#         self.env = env
#         self.state = state
#         self.score = score
#         self.parent = parent
#         self.children = {}
#         self.visits = 0
#         self.total_reward = 0.0
#         self.untried_spawns = self.get_possible_spawns()

#     def fully_expanded(self):
#         return len(self.untried_spawns) == 0

#     def get_possible_spawns(self):
#         possible_spawns = []
#         empty_cells = list(zip(*np.where(self.state == 0)))
#         for (x, y) in empty_cells:
#             possible_spawns.append((x, y, 2, 0.9))
#             possible_spawns.append((x, y, 4, 0.1))
#         return possible_spawns

# class TD_MCTS: 
#     def __init__(self, env, approximator, iterations=500, exploration_constant=1.41,
#                  rollout_depth=10, gamma=0.99):
#         self.env = env
#         self.approximator = approximator
#         self.iterations = iterations
#         self.c = exploration_constant
#         self.rollout_depth = rollout_depth
#         self.gamma = gamma

#     def create_env_from_state(self, state, score):
#         new_env = copy.deepcopy(self.env)
#         new_env.board = state.copy()
#         new_env.score = score
#         return new_env

#     def select_child_decision(self, node):
#         best_value = -float("inf")
#         best_child = None
        
#         for action, child in node.children.items():
#             if child.visits == 0:
#                 uct_value = float("inf")
#             else:
#                 uct_value = (child.total_reward / child.visits) + \
#                             self.c * math.sqrt(math.log(node.visits) / child.visits)
#             if uct_value > best_value:
#                 best_value = uct_value
#                 best_child = child
                
#         return best_child

#     def select_child_chance(self, node):
#         children = list(node.children.values())
#         keys = list(node.children.keys())
#         probs = [k[3] for k in keys]
#         sum_p = sum(probs)
#         if sum_p <= 0:
#             return None
        
#         normalized = [p / sum_p for p in probs]
#         chosen_index = np.random.choice(range(len(children)), p=normalized)
#         return children[chosen_index]

#     def rollout(self, sim_env, depth):
#         total_reward = 0.0
#         discount = 1.0

#         for _ in range(depth):
#             legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
#             if not legal_moves:
#                 break
#             action = random.choice(legal_moves)
#             prev_score = sim_env.score
#             _, _, done, _ = sim_env.step(action)
#             reward = sim_env.score - prev_score
#             total_reward += discount * reward
#             discount *= self.gamma
#             if done:
#                 return total_reward

#         legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
#         if legal_moves:
#             values = []
#             for a in legal_moves:
#                 sim_copy = copy.deepcopy(sim_env)
#                 afterstate, afterscore, _, _ = sim_copy.step(a, spawn_tile=False)
#                 v = self.approximator.value(afterstate)
#                 values.append(v)
#             total_reward += discount * max(values)
#         return total_reward

#     def backpropagate(self, node, reward):
#         while node is not None:
#             node.visits += 1
#             node.total_reward += (reward - node.total_reward) / node.visits
#             node = node.parent

#     def expand_decision_node(self, decision_node):
#         action = random.choice(decision_node.untried_actions)
#         decision_node.untried_actions.remove(action)
#         sim_env = self.create_env_from_state(decision_node.state, decision_node.score)
#         sim_env.step(action, spawn_tile=False)
        
#         new_chance_node = ChanceNode(
#             env = sim_env,
#             state = sim_env.board.copy(),
#             score = sim_env.score,
#             parent = decision_node
#         )
#         decision_node.children[action] = new_chance_node
#         return new_chance_node

#     def expand_chance_node(self, chance_node):
#         x, y, tile, prob = chance_node.untried_spawns.pop()
#         sim_env = self.create_env_from_state(chance_node.state, chance_node.score)
#         sim_env.board[x, y] = tile
        
#         new_decision_node = DecisionNode(
#             env = sim_env,
#             state = sim_env.board.copy(),
#             score = sim_env.score,
#             parent = chance_node
#         )
#         chance_node.children[(x, y, tile, prob)] = new_decision_node
#         return new_decision_node

#     def run_simulation(self, root):
#         node = root
        
#         # selection
#         while True:
#             if not node.fully_expanded():
#                 break
            
#             if isinstance(node, DecisionNode):
#                 if len(node.children) == 0:
#                     break  
#                 node = self.select_child_decision(node)
#             else:  
#                 if len(node.children) == 0:
#                     break
#                 node = self.select_child_chance(node)
                    
#         # expansion
#         if not node.fully_expanded():
#             if isinstance(node, DecisionNode):
#                 node = self.expand_decision_node(node)
#             else:
#                 node = self.expand_chance_node(node)

#         sim_env = self.create_env_from_state(node.state, node.score)
#         rollout_reward = self.rollout(sim_env, self.rollout_depth)

#         self.backpropagate(node, rollout_reward)

#     def best_action_distribution(self, root):
#         total_visits = sum(child.visits for child in root.children.values())
#         distribution = np.zeros(4)
#         best_visits = -1
#         best_action = None
        
#         for action, child in root.children.items():
#             if total_visits > 0:
#                 distribution[action] = child.visits / total_visits
#             else:
#                 distribution[action] = 0
            
#             if child.visits > best_visits:
#                 best_visits = child.visits
#                 best_action = action
#         return best_action, distribution

# import copy
# import math
# import random
# import numpy as np

# class DecisionNode:
#     def __init__(self, env, state, score, parent=None, action=None):
#         self.env = env
#         self.state = state      # board state
#         self.score = score      # current game score
#         self.parent = parent    
#         self.action = action    # the action that led here
#         self.children = {}      # mapping: action -> ChanceNode
#         self.visits = 0
#         self.total_reward = 0.0
#         # Legal moves are those that are allowed based on the current state of env
#         self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]

#     def fully_expanded(self):
#         return len(self.untried_actions) == 0


# class ChanceNode:
#     def __init__(self, env, state, score, parent=None):
#         self.env = env
#         self.state = state      # board state after the decision
#         self.score = score      # score after the decision
#         self.parent = parent
#         self.children = {}      # mapping: outcome (x, y, tile, prob) -> DecisionNode
#         self.visits = 0
#         self.total_reward = 0.0
#         # All potential spawn events (empty cell locations with either a 2 or a 4)
#         self.untried_outcomes = self.get_possible_outcomes()

#     def fully_expanded(self):
#         return len(self.untried_spawns) == 0

#     def get_possible_outcomes(self):
#         outcomes = []
#         empty_cells = list(zip(*np.where(self.state == 0)))
#         for (x, y) in empty_cells:
#             outcomes.append((x, y, 2, 0.9))
#             outcomes.append((x, y, 4, 0.1))
#         return outcomes


# class TD_MCTS:
#     def __init__(self, env, approximator, iterations=500, exploration_constant=1.41,
#                  rollout_depth=10, gamma=0.99):
#         self.env = env
#         self.approximator = approximator
#         self.iterations = iterations
#         self.c = exploration_constant
#         self.rollout_depth = rollout_depth
#         self.gamma = gamma

#     def create_env_from_state(self, state, score):
#         new_env = copy.deepcopy(self.env)
#         new_env.board = state.copy()
#         new_env.score = score
#         return new_env

#     def select_child_decision(self, node):
#         best_value = -float("inf")
#         best_child = None
#         for action, child in node.children.items():
#             # If a node has not been visited, give it the highest possible selection value
#             if child.visits == 0:
#                 uct_value = float("inf")
#             else:
#                 uct_value = (child.total_reward / child.visits) + \
#                             self.c * math.sqrt(math.log(node.visits) / child.visits)
#             if uct_value > best_value:
#                 best_value = uct_value
#                 best_child = child
#         return best_child

#     def select_child_chance(self, node):
#         children = list(node.children.values())
#         keys = list(node.children.keys())
#         # Use the probabilities stored in the outcome tuple (the 4th element)
#         probs = [key[3] for key in keys]
#         sum_p = sum(probs)
#         if sum_p <= 0:
#             return None
        
#         normalized = [p / sum_p for p in probs]
#         chosen_index = np.random.choice(range(len(children)), p=normalized)
#         return children[chosen_index]

#     def rollout(self, sim_env):
#         discount = 1.0
#         total_reward = 0.0
#         # with one-step rollout
#         legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
#         if legal_moves:
#             values = []
#             for a in legal_moves:
#                 sim_copy = copy.deepcopy(sim_env)
#                 afterstate, _, _, _ = sim_copy.step(a, spawn_tile=False)
#                 v = self.approximator.value(afterstate)
#                 values.append(v)
#             total_reward = discount * max(values)
#         return total_reward

#     def backpropagate(self, node, reward):
#         while node is not None:
#             node.visits += 1
#             # Incremental update of the average reward
#             node.total_reward += (reward - node.total_reward) / node.visits
#             node = node.parent

#     def expand_decision_node(self, decision_node):
#         # Select one untried action uniformly at random and remove it
#         action = random.choice(decision_node.untried_actions)
#         decision_node.untried_actions.remove(action)
#         sim_env = self.create_env_from_state(decision_node.state, decision_node.score)
#         # Execute the chosen action without spawning a new tile
#         sim_env.step(action, spawn_tile=False)
#         new_chance_node = ChanceNode(
#             env = sim_env,
#             state = sim_env.board.copy(),
#             score = sim_env.score,
#             parent = decision_node
#         )
#         decision_node.children[action] = new_chance_node
#         return new_chance_node

#     def expand_chance_node(self, chance_node):
#         # Choose one outcome (spawn) from the available untried spawns
#         x, y, tile, prob = chance_node.untried_spawns.pop()
#         sim_env = self.create_env_from_state(chance_node.state, chance_node.score)
#         sim_env.board[x, y] = tile
#         new_decision_node = DecisionNode(
#             env = sim_env,
#             state = sim_env.board.copy(),
#             score = sim_env.score,
#             parent = chance_node
#         )
#         chance_node.children[(x, y, tile, prob)] = new_decision_node
#         return new_decision_node

#     def run_simulation(self, root):
#         node = root

#         # SELECTION PHASE:
#         # Traverse the tree until finding a node that is not fully expanded or has no children.
#         while True:
#             if not node.fully_expanded():
#                 break

#             if isinstance(node, DecisionNode):
#                 if len(node.children) == 0:
#                     break
#                 node = self.select_child_decision(node)
#             else:  # ChanceNode
#                 if len(node.children) == 0:
#                     break
#                 node = self.select_child_chance(node)

#         # EXPANSION PHASE:
#         if not node.fully_expanded():
#             if isinstance(node, DecisionNode):
#                 node = self.expand_decision_node(node)
#             else:
#                 node = self.expand_chance_node(node)

#         # ROLLOUT PHASE:
#         sim_env = self.create_env_from_state(node.state, node.score)
#         rollout_reward = self.rollout(sim_env)

#         # BACKPROPAGATION PHASE:
#         self.backpropagate(node, rollout_reward)

#     def best_action_distribution(self, root):
#         total_visits = sum(child.visits for child in root.children.values())
#         distribution = np.zeros(4)
#         best_visits = -1
#         best_action = None

#         for action, child in root.children.items():
#             distribution[action] = (child.visits / total_visits) if total_visits > 0 else 0
#             if child.visits > best_visits:
#                 best_visits = child.visits
#                 best_action = action
#         return best_action, distribution

import copy
import math
import random
import numpy as np

class DecisionNode:
    def __init__(self, env, parent=None, action=None):
        # Store a deep copy of the environment (which holds board, score, etc.)
        self.env = copy.deepcopy(env)
        self.parent = parent
        self.action = action  # the move that led here (or outcome from chance)
        self.children = {}    # mapping: move action -> ChanceNode
        self.untried_actions = [a for a in range(4) if self.env.is_move_legal(a)]
        self.visits = 0
        self.value = 0.0    # running average value (similar to total_reward in golden sample)
        self.is_terminal = self.env.is_game_over()

    def fully_expanded(self):
        return len(self.untried_actions) == 0

    def uct_select_child(self, exploration=math.sqrt(2)):
        best_value = -float("inf")
        best_child = None
        for action, child in self.children.items():
            if child.visits == 0:
                uct = float("inf")
            else:
                uct = (child.value) + exploration * math.sqrt(math.log(self.visits) / child.visits)
            if uct > best_value:
                best_value = uct
                best_child = child
        return best_child


class ChanceNode:
    def __init__(self, env, parent, action, reward):
        # Make a deep copy of the environment from which a move was taken.
        self.env = copy.deepcopy(env)
        self.parent = parent
        self.action = action       # the decision action that generated this chance node
        self.reward = reward       # immediate reward from that action
        self.children = {}         # mapping: outcome tuple -> DecisionNode
        self.untried_outcomes = self._get_possible_outcomes()
        self.visits = 0
        self.value = 0.0
        self.is_terminal = self.env.is_game_over() or len(self.untried_outcomes) == 0

    def _get_possible_outcomes(self):
        empty = list(zip(*np.where(self.env.board == 0)))
        outcomes = []
        if empty:
            prob2 = 0.9 / len(empty)
            prob4 = 0.1 / len(empty)
            for (x, y) in empty:
                outcomes.append((x, y, 2, prob2))
                outcomes.append((x, y, 4, prob4))
        return outcomes

    def fully_expanded(self):
        return len(self.untried_outcomes) == 0


class TD_MCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41,
                 rollout_depth=10, gamma=0.99):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def rollout(self, sim_env):
        total_reward = 0.0
        discount = 1.0
        # Perform a one-step rollout
        legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
        if legal_moves:
            values = []
            for a in legal_moves:
                sim_copy = copy.deepcopy(sim_env)
                afterstate, _, _, _ = sim_copy.step(a, spawn_tile=False)
                v = self.approximator.value(afterstate)
                values.append(v)
            total_reward += discount * max(values)
        return total_reward

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += (reward - node.value) / node.visits
            node = node.parent

    def expand_decision_node(self, decision_node):
        action = random.choice(decision_node.untried_actions)
        decision_node.untried_actions.remove(action)

        new_env = copy.deepcopy(decision_node.env)
        prev_score = new_env.score
        new_env.step(action, spawn_tile=False)
        reward = new_env.score - prev_score

        new_chance_node = ChanceNode(new_env, parent=decision_node, action=action, reward=reward)
        decision_node.children[action] = new_chance_node
        return new_chance_node

    def expand_chance_node(self, chance_node):
        outcome = chance_node.untried_outcomes.pop()
        x, y, tile, prob = outcome

        new_env = copy.deepcopy(chance_node.env)
        new_env.board[x, y] = tile

        new_decision_node = DecisionNode(new_env, parent=chance_node, action=outcome)
        chance_node.children[outcome] = new_decision_node
        return new_decision_node

    def run_simulation(self, root):
        node = root

        while True:
            if not node.fully_expanded():
                break
            if isinstance(node, DecisionNode):
                if not node.children:
                    break
                node = node.uct_select_child(self.c)
            else:  # ChanceNode
                if not node.children:
                    break
                children = list(node.children.values())
                keys = list(node.children.keys())  # outcome tuples
                probs = [key[3] for key in keys]
                sum_p = sum(probs)
                if sum_p <= 0:
                    break
                normalized = [p / sum_p for p in probs]
                chosen_index = np.random.choice(range(len(children)), p=normalized)
                node = children[chosen_index]

        # EXPANSION: Expand one new child only.
        if not node.fully_expanded():
            if isinstance(node, DecisionNode):
                node = self.expand_decision_node(node)
            else:
                node = self.expand_chance_node(node)

        # ROLLOUT: Simulate a random playout from the new node.
        sim_env = copy.deepcopy(node.env)
        rollout_reward = self.rollout(sim_env)

        # BACKPROPAGATION: Update all nodes in the current branch.
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None

        for action, child in root.children.items():
            distribution[action] = (child.visits / total_visits) if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action

        return best_action, distribution
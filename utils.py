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
    
    def generate_symmetries2(self, pattern):
        # TODO: Generate 8 symmetrical transformations of the given pattern.
        def rotate90(coords, size=4):
            return [(y, size - 1 - x) for x, y in coords]

        def flip_horizontal(coords, size=4):
            return [(x, size - 1 - y) for x, y in coords]
        
        syms = []
        current = pattern
        for _ in range(4):
            syms.append(current)
            syms.append(flip_horizontal(current))
            current = rotate90(current)
        return syms

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

class DecisionNode:
    def __init__(self, env, parent=None, action=None):
        self.env = copy.deepcopy(env)
        self.parent = parent
        self.action = action
        self.children = {}  # action -> ChanceNode
        self.untried_actions = [a for a in range(4) if self.env.is_move_legal(a)]
        self.visits = 0
        self.value = 0.0
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
                uct = child.value / child.visits + exploration * math.sqrt(math.log(self.visits) / child.visits)
            if uct > best_value:
                best_value = uct
                best_child = child
        return best_child


class ChanceNode:
    def __init__(self, env, parent, action, reward):
        self.env = copy.deepcopy(env)
        self.parent = parent
        self.action = action
        self.reward = reward
        self.children = {}  # (x, y, tile, prob) -> DecisionNode
        self.visits = 0
        self.value = 0.0
        self.untried_outcomes = self._get_possible_outcomes()
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

    def sample_outcome(self):
        outcomes, weights = [], []
        for outcome, child in self.children.items():
            outcomes.append(child)
            weights.append(outcome[3])
        return random.choices(outcomes, weights=weights)[0] if weights else random.choice(outcomes)

class TD_MCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, gamma=0.99):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.gamma = gamma

    def create_env_from_state(self, state, score):
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def expand_decision_node(self, node):
        for action in node.untried_actions:
            env = copy.deepcopy(node.env)
            _, reward, _, _ = env.step(action, spawn_tile=False)
            child = ChanceNode(env, node, action, reward)
            child.value = self.approximator.value(env.board)
            node.children[action] = child
        node.untried_actions = []

    def expand_chance_node(self, node):
        for outcome in node.untried_outcomes:
            x, y, val, _ = outcome
            env = copy.deepcopy(node.env)
            env.board[x, y] = val
            child = DecisionNode(env, parent=node, action=outcome)
            node.children[outcome] = child
        node.untried_outcomes = []

    def backpropagate(self, path, value):
        for node in reversed(path):
            node.visits += 1
            node.value += value

    def run_simulation(self, root):
        node = root
        path = [node]
        cumulative_reward = 0

        while not node.is_terminal:
            if isinstance(node, DecisionNode):
                if node.untried_actions:
                    self.expand_decision_node(node)
                node = node.uct_select_child(self.c)
                path.append(node)
                cumulative_reward += node.reward
            elif isinstance(node, ChanceNode):
                if node.untried_outcomes:
                    self.expand_chance_node(node)
                node = node.sample_outcome()
                path.append(node)

            # Stop at leaf
            if (isinstance(node, DecisionNode) and (node.is_terminal or not node.children)) or (isinstance(node, ChanceNode) and not node.children):
                break

        value = self.evaluate_leaf(node, cumulative_reward)
        self.backpropagate(path, value)

    def evaluate_leaf(self, node, cumulative_reward):
        if isinstance(node, DecisionNode):
            if not node.children and not node.is_terminal:
                self.expand_decision_node(node)
            if node.is_terminal or not node.children:
                node_value = 0
            else:
                node_value = max(c.reward + c.value for c in node.children.values())
        elif isinstance(node, ChanceNode):
            node_value = node.value
        return (cumulative_reward + node_value) / 20000  # your original normalization

    def best_action_distribution(self, root):
        dist = np.zeros(4)
        total = sum(c.visits for c in root.children.values())
        best_action = None
        best_visits = -1
        for action, child in root.children.items():
            dist[action] = child.visits / total if total > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action, dist
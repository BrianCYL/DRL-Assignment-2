import math
import random
from collections import defaultdict

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

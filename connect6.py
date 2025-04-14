import sys
import numpy as np
import random
import math
from collections import defaultdict


# (dr, dc) offsets from a given anchor (r, c)
# HORIZONTAL_6 = [(0, i) for i in range(6)]  # A line of 6 cells to the right
# VERTICAL_6   = [(i, 0) for i in range(6)]  # A vertical line downward
# DIAGONAL_6   = [(i, i) for i in range(6)]  # Top-left to bottom-right
# ANTI_DIAG_6  = [(i, -i) for i in range(6)] # Top-right to bottom-left

# OPEN_4_SKIP1 = [(0, 0), (0, 1), (0, 2), (0, 4), (0, 5)]  # Skip (0, 3)
# L_SHAPE = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
# CROSS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]

# OPEN_3 = [(0, i) for i in range(5)]  # Horizontal open-3
# DOUBLE_THREAT = [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)]  # Cross-shaped threat



def rotate_pattern(pattern):
    return [(c, -r) for r, c in pattern]  # 90° clockwise

def flip_horizontal(pattern):
    return [(r, -c) for r, c in pattern]

def flip_vertical(pattern):
    return [(-r, c) for r, c in pattern]

def generate_symmetries(pattern):
    """Generate all 8 symmetrical variants (4 rotations × 2 flips)"""
    variants = set()
    current = pattern
    for _ in range(4):
        current = rotate_pattern(current)
        variants.add(tuple(sorted(current)))
        variants.add(tuple(sorted(flip_horizontal(current))))
        variants.add(tuple(sorted(flip_vertical(current))))
    return [list(p) for p in variants]

def initialize_patterns_with_symmetries():
    raw_patterns = {
        "HORIZONTAL_6": ([(0, i) for i in range(6)], 1000),
        "VERTICAL_6": ([(i, 0) for i in range(6)], 1000),
        "DIAGONAL_6": ([(i, i) for i in range(6)], 10000),
        "ANTI_DIAG_6": ([(i, -i) for i in range(6)], 1000),
        "DOUBLE_THREAT": ([(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)], 500),
        "OPEN_4_SKIP1": ([(0, 0), (0, 1), (0, 2), (0, 4), (0, 5)], 300),
        "OPEN_3": ([(0, i) for i in range(5)], 150),
        "CROSS": ([(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)], 100),
        "L_SHAPE": ([(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)], 75),
        "VERTICAL_5": ([(i, 0) for i in range(5)], 50)
    }

    all_patterns = []
    pattern_values = {}

    for name, (base_pattern, value) in raw_patterns.items():
        symmetries = generate_symmetries(base_pattern)
        for sym in symmetries:
            all_patterns.append(sym)
            # Define a synthetic "ideal" filled pattern to initialize values
            key = tuple([1 if i % 2 != 0 else 0 for i in range(len(sym))])  # e.g. [0,1,1,1,0] or [1,1,1,1,1]
            pattern_values[tuple(key)] = value

    return all_patterns, pattern_values

class ValueApproximator:
    def __init__(self, board_size, patterns):
        self.board_size = board_size
        self.patterns = patterns
        self.values = defaultdict(float)  # key: (tuple of stones), value: scalar

    def extract_pattern(self, board, player, anchor, pattern):
        """Extract pattern as a tuple of relative values from anchor (0: empty, 1: own, 2: opp)"""
        r0, c0 = anchor
        extracted = []
        for dr, dc in pattern:
            r, c = r0 + dr, c0 + dc
            if 0 <= r < self.board_size and 0 <= c < self.board_size:
                val = board[r, c]
                if val == player:
                    extracted.append(1)
                elif val == 0:
                    extracted.append(0)
                else:
                    extracted.append(2)
            else:
                extracted.append(-1)  # Out of bounds
        return tuple(extracted)

    def value(self, board, player):
        total = 0.0
        for r in range(self.board_size):
            for c in range(self.board_size):
                for pattern in self.patterns:
                    key = self.extract_pattern(board, player, (r, c), pattern)
                    total += self.values[key]
        return total

class Node:
    def __init__(self, board, player, parent=None, move=None):
        self.board = np.copy(board)
        self.player = player
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value_sum = 0.0
        self.untried_moves = self.get_legal_moves()

    def q_value(self):
        return 0.0 if self.visits == 0 else self.value_sum / self.visits

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, c=1.4):
        def ucb_score(child):
            return child.q_value() + c * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-4))
        return max(self.children, key=ucb_score)

    def get_legal_moves(self):
        return [(r, c) for r in range(self.board.shape[0]) for c in range(self.board.shape[1]) if self.board[r, c] == 0]

    def expand(self):
        move = self.untried_moves.pop()
        new_board = np.copy(self.board)
        new_board[move] = self.player
        next_player = 3 - self.player
        child = Node(new_board, next_player, self, move)
        self.children.append(child)
        return child
    
class MCTS:
    def __init__(self, value_approximator, rollout_limit=100, c=1.4):
        self.value_fn = value_approximator
        self.rollout_limit = rollout_limit
        self.c = c

    def simulate(self, root_node):
        for _ in range(self.rollout_limit):
            node = root_node
            path = [node]

            # Selection
            while node.is_fully_expanded() and node.children:
                node = node.best_child(self.c)
                path.append(node)

            # Expansion
            if not self.is_terminal(node.board):
                node = node.expand()
                path.append(node)

            # Evaluation using value approximator
            value = self.value_fn.value(node.board, node.player)

            # Backpropagation
            for n in reversed(path):
                n.visits += 1
                # value is always from root's perspective
                n.value_sum += value if n.player == root_node.player else -value

    def is_terminal(self, board):
        temp_env = Connect6Game()
        temp_env.board = np.copy(board)
        return temp_env.check_win() != 0 or np.count_nonzero(board) == board.size

class Connect6Game:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0: Empty, 1: Black, 2: White
        self.turn = 1  # 1: Black, 2: White
        self.game_over = False

    def reset_board(self):
        """Clears the board and resets the game."""
        self.board.fill(0)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)

    def set_board_size(self, size):
        """Sets the board size and resets the game."""
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)

    def check_win(self):
        """Checks if a player has won.
        Returns:
        0 - No winner yet
        1 - Black wins
        2 - White wins
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    current_color = self.board[r, c]
                    for dr, dc in directions:
                        prev_r, prev_c = r - dr, c - dc
                        if 0 <= prev_r < self.size and 0 <= prev_c < self.size and self.board[prev_r, prev_c] == current_color:
                            continue
                        count = 0
                        rr, cc = r, c
                        while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr, cc] == current_color:
                            count += 1
                            rr += dr
                            cc += dc
                        if count >= 6:
                            return current_color
        return 0

    def index_to_label(self, col):
        """Converts column index to letter (skipping 'I')."""
        return chr(ord('A') + col + (1 if col >= 8 else 0))  # Skips 'I'

    def label_to_index(self, col_char):
        """Converts letter to column index (accounting for missing 'I')."""
        col_char = col_char.upper()
        if col_char >= 'J':  # 'I' is skipped
            return ord(col_char) - ord('A') - 1
        else:
            return ord(col_char) - ord('A')

    def play_move(self, color, move):
        """Places stones and checks the game status."""
        if self.game_over:
            print("? Game over")
            return

        stones = move.split(',')
        positions = []

        for stone in stones:
            stone = stone.strip()
            if len(stone) < 2:
                print("? Invalid format")
                return
            col_char = stone[0].upper()
            if not col_char.isalpha():
                print("? Invalid format")
                return
            col = self.label_to_index(col_char)
            try:
                row = int(stone[1:]) - 1
            except ValueError:
                print("? Invalid format")
                return
            if not (0 <= row < self.size and 0 <= col < self.size):
                print("? Move out of board range")
                return
            if self.board[row, col] != 0:
                print("? Position already occupied")
                return
            positions.append((row, col))

        for row, col in positions:
            self.board[row, col] = 1 if color.upper() == 'B' else 2

        self.turn = 3 - self.turn
        print('= ', end='', flush=True)

    def generate_move(self, color):
        """Generates a random move for the computer."""
        if self.game_over:
            print("? Game over")
            return

        # empty_positions = [(r, c) for r in range(self.size) for c in range(self.size) if self.board[r, c] == 0]
        # selected = random.sample(empty_positions, 1)
        # move_str = ",".join(f"{self.index_to_label(c)}{r+1}" for r, c in selected)
        
        # self.play_move(color, move_str)

        # print(f"{move_str}\n\n", end='', flush=True)
        # print(move_str, file=sys.stderr)
        base_patterns = {
        "HORIZONTAL_6": ([(0, i) for i in range(6)], 1000),
        "VERTICAL_6": ([(i, 0) for i in range(6)], 1000),
        "DIAGONAL_6": ([(i, i) for i in range(6)], 1000),
        "ANTI_DIAG_6": ([(i, -i) for i in range(6)], 1000),
        "DOUBLE_THREAT": ([(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)], 500),
        "OPEN_4_SKIP1": ([(0, 0), (0, 1), (0, 2), (0, 4), (0, 5)], 300),
        "OPEN_3": ([(0, i) for i in range(5)], 150),
        "CROSS": ([(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)], 100),
        "L_SHAPE": ([(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)], 75),
        }
        player = 1 if color.upper() == 'B' else 2
        is_first_turn = np.count_nonzero(self.board) < 2
        patterns = []
        pattern_values = {}
        for name, (pattern, val) in base_patterns.items():
            patterns.append(pattern)
            pattern_values[tuple([1]*len(pattern))] = val  # assume fully filled patterns for now

        # Initialize approximator and MCTS
        approximator = ValueApproximator(board_size=self.size, patterns=patterns, initial_values=pattern_values)
        mcts = MCTS(value_fn=approximator, rollout_limit=100)
        root = Node(np.copy(self.board), player)
        mcts.simulate(root)

        # Choose 1 move (first turn) or 2 best moves
        best_children = sorted(root.children, key=lambda c: c.visits, reverse=True)
        if is_first_turn:
            selected_moves = [best_children[0].move]
        else:
            selected_moves = [c.move for c in best_children[:2]]

        move_str = ",".join(f"{self.index_to_label(c)}{r+1}" for r, c in selected_moves)
        self.play_move(color, move_str)
        print(f"{move_str}\n\n", end='', flush=True)

    def show_board(self):
        """Displays the board as text."""
        print("= ")
        for row in range(self.size - 1, -1, -1):
            line = f"{row+1:2} " + " ".join("X" if self.board[row, col] == 1 else "O" if self.board[row, col] == 2 else "." for col in range(self.size))
            print(line)
        col_labels = "   " + " ".join(self.index_to_label(i) for i in range(self.size))
        print(col_labels)
        print(flush=True)

    def list_commands(self):
        """Lists all available commands."""
        print("= ", flush=True)  

    def process_command(self, command):
        """Parses and executes GTP commands."""
        command = command.strip()
        if command == "get_conf_str env_board_size:":
            return "env_board_size=19"

        if not command:
            return
        
        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "boardsize":
            try:
                size = int(parts[1])
                self.set_board_size(size)
            except ValueError:
                print("? Invalid board size")
        elif cmd == "clear_board":
            self.reset_board()
        elif cmd == "play":
            if len(parts) < 3:
                print("? Invalid play command format")
            else:
                self.play_move(parts[1], parts[2])
                print('', flush=True)
        elif cmd == "genmove":
            if len(parts) < 2:
                print("? Invalid genmove command format")
            else:
                self.generate_move(parts[1])
        elif cmd == "showboard":
            self.show_board()
        elif cmd == "list_commands":
            self.list_commands()
        elif cmd == "quit":
            print("= ", flush=True)
            sys.exit(0)
        else:
            print("? Unsupported command")

    def run(self):
        """Main loop that reads GTP commands from standard input."""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                self.process_command(line)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"? Error: {str(e)}")



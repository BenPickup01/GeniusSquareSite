import numpy as np 
from genius_square_methods import apply_block_transformations
import random
from tqdm import tqdm
from collections import defaultdict

class grid_space:
    def __init__(self, ):
        self.board = np.zeros(shape=(6,6), dtype=np.float32)

    def reset_board(self, test_dice=None, dice_set="testing"):
        # Reset the state to a random genius square configuration 
        dice_sets = {
            "testing": [
            [[0, 0], [2, 0], [3, 0], [3, 1], [4, 1], [5, 2]],  # Die 1
            [[0, 1], [1, 1], [2, 1], [0, 2], [1, 0], [1, 2]],  # Die 2
            [[2, 2], [3, 2], [4, 2], [1, 3], [2, 3], [3, 3]],  # Die 3
            [[4, 0], [5, 1], [1, 5], [0, 4]],                   # Die 4
            [[0, 3], [1, 4], [2, 5], [2, 4], [3, 5], [5, 5]],  # Die 5
            [[4, 3], [5, 3], [4, 4], [5, 4], [3, 4], [4, 5]],  # Die 6
            [[5, 0], [0, 5]]                                   # Die 7
        ], 
            "training": [
            [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
            [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3]],
            [[3, 2], [3, 4], [4, 3], [4, 5], [5, 4]],
            [[3, 0], [4, 0], [4, 1], [5, 0], [5, 1]],
            [[0, 4], [0, 5], [1, 4], [1, 5], [2, 5]],
            [[2, 0], [3, 1], [4, 2], [5, 2], [5, 3]],
            [[0, 2], [0, 3], [1, 3], [2, 4], [3, 5]]
            ]
        }
        
        self.board = np.zeros((6,6), dtype=np.float32) # Defines an empty grid
        game_die = dice_sets[dice_set] 
        if test_dice is None:
            # Fills in the blocker pieces as 1 in the observation space to identify that they have been filed
            for dice in game_die:
                choice = random.choice(dice)
                self.board[choice[0], choice[1]] = 1
        else:
            for dice in test_dice:
                self.board[dice[0], dice[1]] = 1

    def show_grid(self):
        for i in self.board:
            print(i)
    
    def get_future_states(self, block_id):
        """Compute all valid placements for the current block"""
        valid_actions = []
        future_states = []
        # Define rotation and inversion limits per block type
        block_transformations = {
            0: (1, 1),  # 2x2 Square piece 
            1: (2, 1),  # 4x1 Long Piece
            2: (4, 1),  # T Piece  
            3: (2, 2),  # Z Piece   
            4: (4, 2),  # L Piece
            5: (4, 1),  # Mini L   
            6: (2, 1),  # 3 Long 
            7: (2, 1),  # 2 Long
            8: (1, 1),  # 1x1
        }
        
        max_rotations, max_inversions = block_transformations.get(block_id)
        
        for x in range(6):
            for y in range(6):
                for rot in range(max_rotations):
                    for inv in range(max_inversions):
                        transformed_block = apply_block_transformations((rot, inv), block_id)

                        flag, board = self.is_valid_placement(transformed_block, (x, y))
                        if flag:
                            valid_actions.append((x, y, rot, inv))
                            future_states.append(board)
        
        return future_states

    def is_valid_placement(self, block, position):
        """Check if a block can be placed at a given position"""
        x, y = position
        rows, cols = block.shape
        
        if x + rows > 6 or y + cols > 6:  # Out-of-bounds check
            return False, []
        
        for i in range(rows):
            for j in range(cols):
                if block[i, j] == 1 and self.board[x + i, y + j] == 1:
                    return False, []  # Overlapping check
        
        # return copy of that grid with the block placed
        output_board = self.board.copy()
        for i in range(rows):
            for j in range(cols):
                if block[i, j] == 1:
                    output_board[x + i, y + j] = 1

        return True, output_board
    
    def set_state(self, grid):
        self.board = grid
    
def determine_block_order(num_trials=1000, dice_set="testing", max_steps=9):
    """
    Simulate many games to determine the best block order based on fewest average possible moves.
    
    Args:
        num_trials (int): Number of trials to run.
        dice_set (str): Dice set to use ("testing" or "training").
        max_steps (int): Maximum number of blocks to place (default 9 for all blocks).
    
    Returns:
        list: The most common block order.
    """
    block_orders = []
    block_names = {
        0: "2x2 Square",
        1: "4x1 Long",
        2: "T Piece",
        3: "Z Piece",
        4: "L Piece",
        5: "Mini L",
        6: "3 Long",
        7: "2 Long",
        8: "1x1"
    }

    for trial in tqdm(range(num_trials), desc="Simulating games"):
        game_board = grid_space()
        game_board.reset_board(dice_set=dice_set)
        remaining_blocks = list(range(9))  # Blocks 0 to 8
        current_order = []
        step = 0

        while remaining_blocks and step < max_steps:
            # Compute average number of possible moves for each remaining block
            move_counts = {}
            for block_id in remaining_blocks:
                future_states = game_board.get_future_states(block_id)
                move_counts[block_id] = len(future_states)

            # Select block with fewest possible moves
            min_moves = float('inf')
            chosen_block = None
            for block_id, count in move_counts.items():
                if count > 0 and count < min_moves:
                    min_moves = count
                    chosen_block = block_id
                elif count > 0 and count == min_moves and chosen_block is not None:
                    # Break ties randomly
                    if random.random() < 0.5:
                        chosen_block = block_id

            if chosen_block is None:
                # No valid moves for any remaining block
                break

            current_order.append(chosen_block)
            remaining_blocks.remove(chosen_block)

            # Place a random valid placement for the chosen block
            future_states = game_board.get_future_states(chosen_block)
            if future_states:
                chosen_state = random.choice(future_states)
                game_board.set_state(chosen_state)
            else:
                break

            step += 1

        block_orders.append(current_order)

    # Compute the most common block order
    # For simplicity, we take the first complete order (with all 9 blocks) or average the orders
    complete_orders = [order for order in block_orders if len(order) == max_steps]
    if complete_orders:
        # Return the first complete order as a representative
        representative_order = complete_orders[0]
        print("\nRepresentative block order:")
        for i, block_id in enumerate(representative_order):
            print(f"Step {i+1}: {block_names[block_id]}")
        return representative_order
    else:
        # Fallback: Average the partial orders
        print("\nNo complete orders found. Using partial order from first trial.")
        representative_order = block_orders[0]
        for i, block_id in enumerate(representative_order):
            print(f"Step {i+1}: {block_names[block_id]}")
        return representative_order

def determine_worst_block_order(num_trials=1000, dice_set="testing", max_steps=9):
    """
    Simulate many games to determine the worst block order based on the highest average possible moves.
    
    Args:
        num_trials (int): Number of trials to run.
        dice_set (str): Dice set to use ("testing" or "training").
        max_steps (int): Maximum number of blocks to place (default 9 for all blocks).
    
    Returns:
        list: The worst block order (blocks with most possible moves first).
    """
    block_orders = []
    block_names = {
        0: "2x2 Square",
        1: "4x1 Long",
        2: "T Piece",
        3: "Z Piece",
        4: "L Piece",
        5: "Mini L",
        6: "3 Long",
        7: "2 Long",
        8: "1x1"
    }

    for trial in tqdm(range(num_trials), desc="Simulating games for worst order"):
        game_board = grid_space()
        game_board.reset_board(dice_set=dice_set)
        remaining_blocks = list(range(9))  # Blocks 0 to 8
        current_order = []
        step = 0

        while remaining_blocks and step < max_steps:
            # Compute number of possible moves for each remaining block
            move_counts = {}
            for block_id in remaining_blocks:
                future_states = game_board.get_future_states(block_id)
                move_counts[block_id] = len(future_states)

            # Select block with the most possible moves
            max_moves = -1
            chosen_block = None
            for block_id, count in move_counts.items():
                if count > max_moves:
                    max_moves = count
                    chosen_block = block_id
                elif count == max_moves and chosen_block is not None:
                    # Break ties randomly
                    if random.random() < 0.5:
                        chosen_block = block_id

            if chosen_block is None:
                # No valid moves for any remaining block
                break

            current_order.append(chosen_block)
            remaining_blocks.remove(chosen_block)

            # Place a random valid placement for the chosen block
            future_states = game_board.get_future_states(chosen_block)
            if future_states:
                chosen_state = random.choice(future_states)
                game_board.set_state(chosen_state)
            else:
                break

            step += 1

        block_orders.append(current_order)

    # Compute the representative worst block order
    complete_orders = [order for order in block_orders if len(order) == max_steps]
    if complete_orders:
        # Return the first complete order as a representative
        representative_order = complete_orders[0]
        print("\nRepresentative worst block order:")
        for i, block_id in enumerate(representative_order):
            print(f"Step {i+1}: {block_names[block_id]}")
        return representative_order
    else:
        # Fallback: Use the first trial's order
        print("\nNo complete orders found. Using partial order from first trial.")
        representative_order = block_orders[0]
        for i, block_id in enumerate(representative_order):
            print(f"Step {i+1}: {block_names[block_id]}")
        return representative_order
    
def evaluate_block_order(block_order, num_trials=1000, dice_set="testing"):
    """
    Simulate games using a specified block order and return the average game length.
    
    Args:
        block_order (list): List of block IDs (0-8) specifying the order to place blocks.
        num_trials (int): Number of trials to run.
        dice_set (str): Dice set to use ("testing" or "training").
    
    Returns:
        float: Average game length (number of blocks placed).
    """
    if not block_order or not all(0 <= x <= 8 for x in block_order):
        raise ValueError("Invalid block order. Must be a list of integers from 0 to 8.")
    
    game_lengths = []
    block_names = {
        0: "2x2 Square",
        1: "4x1 Long",
        2: "T Piece",
        3: "Z Piece",
        4: "L Piece",
        5: "3 Long",
        6: "Mini L",
        7: "2 Long",
        8: "1x1"
    }

    for trial in tqdm(range(num_trials), desc="Evaluating block order"):
        game_board = grid_space()
        game_board.reset_board(dice_set=dice_set)
        steps = 0

        # Follow the block order
        for block_id in block_order:
            future_states = game_board.get_future_states(block_id)
            if not future_states:
                # No valid placement for this block, game ends
                break
            
            # Place a random valid placement
            chosen_state = random.choice(future_states)
            game_board.set_state(chosen_state)
            steps += 1

        game_lengths.append(steps)

    # Compute statistics
    avg_length = sum(game_lengths) / len(game_lengths)
    success_rate = sum(1 for length in game_lengths if length == len(block_order)) / num_trials * 100
    length_counts = defaultdict(int)
    for length in game_lengths:
        length_counts[length] += 1

    # Print results
    print("\nEvaluation Results for Block Order:")
    for i, block_id in enumerate(block_order):
        print(f"Step {i+1}: {block_names[block_id]}")
    print(f"\nAverage Game Length: {avg_length:.2f} blocks")
    print(f"Success Rate (full order completed): {success_rate:.2f}%")
    print("Distribution of Game Lengths:")
    for length in sorted(length_counts.keys()):
        print(f"  {length} blocks: {length_counts[length]} times ({length_counts[length]/num_trials*100:.2f}%)")

# if __name__ == "__main__":
#     # Note: apply_block_transformations is not provided, so this code assumes it exists
#     # Run the simulation for the worst block order
#     best_order = determine_block_order(num_trials=10000, dice_set="testing")
#     print("\nFinal Best block order IDs:", best_order)

#     print("Evaluating Best Order:")
#     avg_length_best = evaluate_block_order(best_order, num_trials=100000, dice_set="testing")

if __name__ == "__main__":
    # Example block orders (from best and worst, assuming from previous functions)
    best_order = [0, 1, 3, 2, 4, 5, 6, 7, 8] # Example: L, T, 2x2, Z, 4x1, Mini L, 3 Long, 2 Long, 1x1
    worst_order = [4, 5, 7, 8, 6, 3, 0, 1, 2]  # Example: 1x1, 2 Long, 3 Long, Mini L, 4x1, Z, 2x2, T, L


    print("Evaluating Best Order:")
    avg_length_best = evaluate_block_order(worst_order, num_trials=10000, dice_set="testing")

    # print("\nEvaluating Worst Order:")
    # avg_length_worst = evaluate_block_order(worst_order, num_trials=100000, dice_set="testing")



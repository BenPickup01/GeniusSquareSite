import numpy as np


BLOCKS = [
    np.array([[1, 1], [1, 1]]),         # 2x2
    np.array([[1, 1, 1, 1]]),           # 1x4
    np.array([[1, 1, 1],[0, 1, 0]]),    # T Piece
    np.array([[1, 0],[1, 1],[0, 1]]),   # Z Piece
    np.array([[1, 0],[1, 0],[1, 1]]),   # L Piece
    np.array([[1, 0],[1, 1]]),          # Mini L piece
    np.array([[1, 1, 1]]),              # 3x1
    np.array([[1, 1]]),                 # 2x1
    np.array([[1]]),                    # 1x1

]

# Make the code above more elegant and readable by using numpy functions and removing the for loops
def apply_block_transformations(action, block_id):
    no_rotations = action[0]
    invert = action[1]
    
    # Use a copy to avoid modifying the original block
    selected_piece = BLOCKS[block_id].copy()
    
    # Rotate the block
    selected_piece = np.rot90(selected_piece, k=no_rotations)
    
    # Invert the block if required
    if invert == 1:
        selected_piece = np.flip(selected_piece, axis=1)  # vertical flip (x-axis)


    return selected_piece

if __name__ == "__main__":
    # Example usage
    action = (0, 1)  # Rotate 90 degrees and invert
    block_id = 4  # Choose a block ID
    transformed_block = apply_block_transformations(action, block_id)
    print(transformed_block)
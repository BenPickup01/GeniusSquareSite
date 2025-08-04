import numpy as np

def get_rows_score(board):
        filled_rows = np.sum(np.all(board == 1, axis=1))
        filled_columns = np.sum(np.all(board == 1, axis=0))

        return (filled_rows + filled_columns) / 12

def get_largest_empty_space(grid, empty_value=0):

    rows, cols = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    max_size = 0
    isolated_monos = 0

    def dfs(r, c):
        """Perform DFS to compute the size of the connected component."""
        stack = [(r, c)]
        size = 0
        while stack:
            i, j = stack.pop()
            # Check boundaries and visited flag.
            if i < 0 or i >= rows or j < 0 or j >= cols:
                continue
            if visited[i, j]:
                continue
            if grid[i, j] != empty_value:
                continue
            visited[i, j] = True
            size += 1
            # Add 4-connected neighbors (up, down, left, right)
            stack.extend([(i-1, j), (i+1, j), (i, j-1), (i, j+1)])
        return size

    # Iterate over every cell to start a DFS when we find an unvisited empty cell.
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == empty_value and not visited[i, j]:
                region_size = dfs(i, j)
                if region_size > max_size:
                    max_size = region_size
                if region_size == 1:
                    isolated_monos += 1

    return max_size, isolated_monos
import numpy as np

import numpy as np

def find_monospaces(grid):
    """
    Given a 6×6 array of 0s and 1s, returns a new 6×6 array
    where positions of “mono spaces” (a 0 whose four orthogonal
    neighbors are all 1, with out‑of‑bounds considered as 1)
    are marked 1, and all others 0.
    """
    grid = np.asarray(grid)
    if grid.shape != (6, 6):
        raise ValueError("Input must be a 6×6 array.")
    
    # Pad on all sides with 1s so edges automatically see 'filled' beyond boundary
    padded = np.pad(grid, pad_width=1, mode='constant', constant_values=1)
    
    monospaces = np.zeros_like(grid)
    
    for i in range(6):
        for j in range(6):
            if grid[i, j] == 0:
                # orthogonal neighbors in the padded array:
                up    = padded[i+0, j+1]
                down  = padded[i+2, j+1]
                left  = padded[i+1, j+0]
                right = padded[i+1, j+2]
                
                if (up == 1 and down == 1 and left == 1 and right == 1):
                    monospaces[i, j] = 1
    
    return monospaces


if __name__ == "__main__":
    test_grid = np.array([
        [1, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 1],
        [1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])

    print(find_monospaces(test_grid))
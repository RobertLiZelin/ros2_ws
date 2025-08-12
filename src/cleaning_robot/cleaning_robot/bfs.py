import numpy as np
from collections import deque
from .utils import is_cell_traversable

def bfs_nearest_unclean(grid, start_r, start_c, robot_radius_cells=6):
    """
    Breadth-First Search for the nearest reachable 'unclean' cell.

    This function performs a BFS starting from (start_r, start_c) to find the
    closest grid cell that:
      1) is marked as 'unclean/free' in your map convention (grid[r, c] == 1), and
      2) is feasible for the robot footprint (checked via `is_cell_traversable`).

    The search expands in 4-connected space (up, down, left, right).
    It returns as soon as it encounters the first valid target, which guarantees
    minimal number of grid steps from the start (in an unweighted grid).

    Args:
        grid (np.ndarray):
            2D occupancy-like grid of shape (rows, cols). Project convention:
              - 0: obstacle / blocked
              - 1: free / unclean (i.e., traversable and yet to be cleaned)
              - -1: unknown (typically treated as non-traversable)
        start_r (int): Start row index.
        start_c (int): Start column index.
        robot_radius_cells (int, optional):
            Robot radius expressed in number of grid cells. Used by
            `is_cell_traversable` to ensure the footprint can safely occupy a cell.
            Defaults to 6 (larger value -> more conservative reachability).

    Returns:
        tuple[int, int] | None:
            The target cell (row, col) if found; otherwise None when no
            reachable unclean cell exists.

    Notes:
        - BFS explores in layers of equal distance from the start, so the first
          valid (grid == 1 && traversable) cell found is the nearest in terms of
          Manhattan grid steps.
        - This function does not compute a path; it only returns the goal cell.
          Use A* (or another planner) afterwards to generate the actual path.
        - Make sure (start_r, start_c) is inside the map. If it's outside, the
          queue will still be seeded but immediately discarded by bounds checks.

    Complexity:
        - Time: O(rows * cols) in the worst case (visits each cell at most once).
        - Space: O(rows * cols) for the visited set/array and the queue in the
          worst case.
    """
    rows, cols = grid.shape

    # Track visited cells to avoid re-processing the same locations.
    # Using a boolean array is memory-efficient and fast for random access.
    visited = np.zeros_like(grid, dtype=bool)

    # Standard BFS queue; we push the starting location as the first frontier.
    queue = deque()
    queue.append((start_r, start_c))

    while queue:
        r, c = queue.popleft()

        # Bounds check (defensive). If (r, c) is invalid, skip it.
        if not (0 <= r < rows and 0 <= c < cols):
            continue

        # Skip if already visited. Mark as visited once we pop it to ensure
        # each cell is processed at most once.
        if visited[r, c]:
            continue
        visited[r, c] = True

        # If the current cell is 'unclean/free' AND traversable considering
        # the robot's footprint, we return it as the nearest acceptable goal.
        if grid[r, c] == 1 and is_cell_traversable(grid, r, c, robot_radius_cells):
            return (r, c)

        # Push 4-connected neighbors (up, down, left, right) into the queue
        # if they are within bounds and not yet visited. We do not check
        # traversability here, because BFS is only trying to locate a valid
        # target cell; actual path feasibility to that cell will be handled
        # by the path planner (e.g., A*). The traversability check above
        # ensures the target cell itself can be occupied by the robot.
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                queue.append((nr, nc))

    # No reachable 'unclean' cell was found.
    return None

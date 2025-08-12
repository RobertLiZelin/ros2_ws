import numpy as np
from .utils import is_cell_traversable


def generate_sample_map(rows=80, cols=120, obstacle_ratio=0.001, start=(10, 10),
                        robot_radius_cells=6):
    """
    Generate a simulated occupancy grid map for testing purposes.

    This function is useful when no SLAM map is available (e.g., in simulation mode).
    It creates a mostly empty map with thin obstacle borders around the edges
    and randomly placed obstacles, ensuring that the start position is always navigable.

    Args:
        rows (int): Number of rows (grid height).
        cols (int): Number of columns (grid width).
        obstacle_ratio (float):
            Fraction of available (non-protected) cells to turn into obstacles.
            Example: 0.001 means 0.1% of eligible cells will become obstacles.
        start (tuple[int, int]):
            Starting grid coordinates (row, col) for the robot.
        robot_radius_cells (int):
            Robot's radius in number of grid cells. Used to ensure the start position
            has enough clearance.

    Returns:
        np.ndarray:
            2D integer array of shape `(rows, cols)` representing the map:
            - `1` = free/navigable cell
            - `0` = obstacle cell

    Raises:
        ValueError:
            If the start position cannot accommodate the robot (clearance check fails).

    Notes:
        - Obstacles are placed randomly, except in a protected "buffer zone" around the start.
        - Edges are marked as obstacles (one-cell thick border) to avoid pathing outside.
    """
    # Fixed random seed for reproducibility of generated map
    np.random.seed(42)

    # Initialize map as all free cells (1)
    grid = np.ones((rows, cols), dtype=int)

    # Set map borders as obstacles (0) - 1 cell thick
    grid[0, :] = 0
    grid[-1, :] = 0
    grid[:, 0] = 0
    grid[:, -1] = 0

    # Protect a padded region around the start position from obstacles
    sr, sc = start
    protected = np.zeros_like(grid, dtype=bool)
    padding = robot_radius_cells + 1
    protected[max(0, sr - padding):sr + padding + 1,
              max(0, sc - padding):sc + padding + 1] = True

    # Find all cells eligible for obstacle placement (not protected, currently free)
    free_indices = np.argwhere((grid == 1) & (~protected))

    # Determine number of random obstacles to place
    obstacle_count = int(len(free_indices) * obstacle_ratio)

    # Select random free cell indices for obstacles
    obstacle_indices = free_indices[
        np.random.choice(len(free_indices), obstacle_count, replace=False)
    ]
    for r, c in obstacle_indices:
        grid[r, c] = 0

    # Ensure start position is navigable considering robot's size
    if not is_cell_traversable(grid, sr, sc, robot_radius_cells):
        raise ValueError(
            "⚠️ Start position cannot fit the robot. "
            "Adjust map size, start location, or obstacle placement."
        )

    return grid


def convert_slam_map_to_internal(slam_map, width, height):
    """
    Convert a SLAM-generated OccupancyGrid (1D data array) into the internal grid format.

    Args:
        slam_map (list[int] or np.ndarray):
            Flattened occupancy grid data from `nav_msgs/OccupancyGrid.data`.
            SLAM map uses:
                0   = free space
                100 = obstacle
                -1  = unknown space
        width (int): Width of the map (number of columns).
        height (int): Height of the map (number of rows).

    Returns:
        np.ndarray:
            2D array with:
            - `1` for free/navigable cells
            - `0` for obstacles or unknown cells
    """
    # Convert 1D array to 2D grid of shape (height, width)
    arr = np.array(slam_map).reshape((height, width))
    # Convert free space (0) to 1, all others (100/-1) to 0
    return np.where(arr == 0, 1, 0)

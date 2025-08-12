import numpy as np
import cv2


def is_cell_traversable(grid, r, c, robot_radius_cells):
    """
    Check if a circular area centered at (r, c) is fully traversable for the robot.

    The check ensures that all cells within the robot's radius are free space.

    Args:
        grid (np.ndarray): 2D map array where:
            0 = obstacle
            1 = free/uncleaned
            2 = cleaned
        r (int): Row index of the center cell.
        c (int): Column index of the center cell.
        robot_radius_cells (int): Robot radius in cells.

    Returns:
        bool: True if the robot can fully fit in this location without hitting obstacles.
    """
    rows, cols = grid.shape
    for dr in range(-robot_radius_cells, robot_radius_cells + 1):
        for dc in range(-robot_radius_cells, robot_radius_cells + 1):
            # Only check cells inside the circular footprint
            if dr**2 + dc**2 > robot_radius_cells**2:
                continue
            nr, nc = r + dr, c + dc
            # If out of bounds or hitting an obstacle → not traversable
            if not (0 <= nr < rows and 0 <= nc < cols) or grid[nr, nc] == 0:
                return False
    return True


def compute_obstacle_distance_map(grid):
    """
    Compute a distance map from each free cell to the nearest obstacle.

    Uses OpenCV's `distanceTransform` to measure Euclidean distance (in cells).

    Args:
        grid (np.ndarray): 2D map array.
            0 = obstacle, 1/2 = free/cleaned space

    Returns:
        np.ndarray: Float32 array of distances (same shape as grid), 
                    where each value is the distance (in cells) to the closest obstacle.
    """
    # Create a binary mask: obstacle=1, free=0
    obstacle_mask = (grid == 0).astype(np.uint8)
    # Invert the mask so free=1, obstacle=0, then compute distance transform
    dist = cv2.distanceTransform(1 - obstacle_mask, cv2.DIST_L2, 3)
    return dist


def mark_robot_footprint_clean(grid, r, c, robot_radius_cells):
    """
    Mark the robot's footprint as cleaned (value=2) on the grid.

    Uses an inscribed square instead of a full circle for efficiency.

    Args:
        grid (np.ndarray): Cleaning map.
        r, c (int): Robot's center cell.
        robot_radius_cells (int): Robot radius in cells.
    """
    rows, cols = grid.shape
    # Half-size of the inscribed square inside the robot's circular footprint
    half = int(np.floor(robot_radius_cells / np.sqrt(2)))

    for dr in range(-half, half + 1):
        for dc in range(-half, half + 1):
            nr, nc = r + dr, c + dc
            # Only mark uncleaned cells (value=1)
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 1:
                grid[nr, nc] = 2


def mark_footprint_along_path(grid, start, end, robot_radius_cells):
    """
    Mark all cells along the path from `start` to `end` as cleaned.

    Interpolates between the two points and marks the robot's footprint at each step.

    Args:
        grid (np.ndarray): Cleaning map.
        start (tuple): Starting cell (row, col).
        end (tuple): Ending cell (row, col).
        robot_radius_cells (int): Robot radius in cells.
    """
    r1, c1 = start
    r2, c2 = end

    # Estimate number of steps based on Euclidean distance between start and end
    steps = int(np.hypot(r2 - r1, c2 - c1))
    if steps == 0:
        steps = 1

    # Walk along the path and mark the footprint at each point
    for i in range(steps + 1):
        t = i / steps
        r = int(round(r1 + t * (r2 - r1)))
        c = int(round(c1 + t * (c2 - c1)))
        mark_robot_footprint_clean(grid, r, c, robot_radius_cells)

import heapq
from .utils import is_cell_traversable

def astar(grid, start, goal, robot_radius_cells=1, distance_map=None, alpha=0.5):
    """
    A* path planning over a 2D occupancy grid with simple kinodynamic preferences.

    This implementation:
      - Treats the grid as 4-connected (N, S, W, E).
      - Respects robot footprint by checking traversability with `robot_radius_cells`.
      - Adds a turn penalty to prefer straighter paths.
      - Adds a "hug-the-obstacle" reward using an optional obstacle distance map to
        bias the path closer to walls/borders (useful for cleaning coverage).

    Args:
        grid (np.ndarray): 2D array of shape (rows, cols).
            Convention used by the rest of this project (based on your utils):
              * 0 -> obstacle (blocked)
              * 1 -> free (traversable)
              * -1 -> unknown (treat as non-free unless your utils say otherwise)
        start (tuple[int, int]): Start cell as (row, col).
        goal (tuple[int, int]):  Goal cell as (row, col).
        robot_radius_cells (int): Robot radius expressed in cells. Used by
            `is_cell_traversable` to ensure the footprint can fit.
        distance_map (np.ndarray | None): Optional 2D float array where each cell
            stores the distance to the nearest obstacle (or some monotonic proxy).
            Larger values = farther from obstacles. Smaller values = closer to obstacles.
        alpha (float): Weight for the obstacle-hugging reward. Larger alpha makes
            the planner prefer being closer to obstacles more strongly.

    Returns:
        list[tuple[int, int]]: A list of cells (row, col) from start to goal
        inclusive. Returns an empty list if no path is found.

    Notes on cost shaping:
      - Base step cost is 1 per move.
      - Turn penalty (default 1.5) is added when the direction changes, encouraging
        straighter paths.
      - Distance reward is NEGATIVE cost when near obstacles:
            distance_reward = -alpha / (dist_to_obstacle + 0.1)
        So when `dist_to_obstacle` is small (near an obstacle), reward magnitude
        increases (i.e., total cost decreases). This biases solutions to "hug" walls.
      - Heuristic is Manhattan distance (admissible for 4-connected grids with unit costs).
    """
    rows, cols = grid.shape

    # Priority queue entries are tuples:
    #   (f_score, g_score, current_cell, previous_direction)
    # where:
    #   f_score = g_score + heuristic(current, goal)
    #   previous_direction is one of the 4-neighborhood vectors or None at start
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start, None))

    # came_from: neighbor -> predecessor to reconstruct the path at the end
    came_from = {}

    # g_score: actual cost from start to a given node
    g_score = {start: 0}

    # 4-connected neighborhood: Up, Down, Left, Right (row, col)
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while open_set:
        # Pop the cell with the lowest f_score; tie-broken by g_score
        _, current_g, current, prev_dir = heapq.heappop(open_set)

        # Goal reached -> reconstruct path
        if current == goal:
            return reconstruct_path(came_from, current)

        # Explore neighbors
        for d in dirs:
            neighbor = (current[0] + d[0], current[1] + d[1])

            # Bounds check
            if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
                continue

            # Quick reject blocked cells (0 means obstacle in this codebase convention)
            if grid[neighbor] == 0:
                continue

            # Respect robot footprint: ensure the robot can actually occupy 'neighbor'
            if not is_cell_traversable(grid, neighbor[0], neighbor[1], robot_radius_cells):
                continue

            # Penalize turning to prefer straighter segments
            turn_penalty = 0.0
            if prev_dir is not None and d != prev_dir:
                turn_penalty = 1.5

            # Reward being near obstacles if a distance map is available
            # distance_map: larger means farther from obstacles
            # We invert it so that smaller distance => larger (negative) reward (lower cost)
            distance_reward = 0.0
            if distance_map is not None:
                dist_to_obstacle = distance_map[neighbor]
                # Add small epsilon 0.1 to avoid division by zero and to cap the magnitude
                distance_reward = -alpha / (dist_to_obstacle + 0.1)

            # Unit move cost (grid step) + turn penalty + proximity reward
            tentative_g = current_g + 1 + turn_penalty + distance_reward

            # Standard A* relaxation
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, tentative_g, neighbor, d))

    # No path found
    return []


def heuristic(a, b):
    """
    Manhattan distance heuristic for 4-connected grids.

    Args:
        a (tuple[int, int]): (row, col) of node A.
        b (tuple[int, int]): (row, col) of node B.

    Returns:
        int: |Δrow| + |Δcol|
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def reconstruct_path(came_from, current):
    """
    Reconstruct a path from the 'came_from' map.

    Args:
        came_from (dict): Maps node -> predecessor node.
        current (tuple[int, int]): Goal node.

    Returns:
        list[tuple[int, int]]: Path from start to current (inclusive).
    """
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

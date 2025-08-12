#!/usr/bin/env python3
"""
CleanerNode
-----------
High-level flow:

SLAM (OccupancyGrid on /map)
  -> convert to internal grid for planning
  -> plan path (A* / BFS) in grid coordinates (row, col)
  -> convert to world coordinates (meters) using map origin + resolution
  -> try to send path to Nav2 controller via FollowPath action
     - if available: Nav2 executes the path on the real robot
     - if NOT available: fall back to local simulation loop
       (still publishes map and path for RViz2 visualization)

Key integrations:
- Uses TF (map -> base_link) to fetch robot's live start position.
- Honors OccupancyGrid.info.resolution and .origin (x, y) when converting
  (row, col) -> (x, y) in meters.
- Provides a simple path densification to help smooth controller tracking.

Coordinate conventions:
- Grid indices are (row, col) with row = Y-like index (downwards on array),
  col = X-like index (rightwards on array).
- World coordinates are (x, y) in meters in the "map" frame from SLAM.

Map encoding used internally in this project (by convention):
- grid == 0 : obstacle (not traversable)
- grid == 1 : free AND uncleaned (target space)
- grid == 2 : cleaned (already covered)
- grid == -1: unknown (treated as not-free for planning)
"""

import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from nav_msgs.msg import OccupancyGrid

# Project modules
from .path_executor import PathExecutor           # Wraps Nav2 FollowPath action
from .astar import astar                          # Global planner (A*)
from .bfs import bfs_nearest_unclean              # Nearest-target selection (BFS)
from .map_manager import (
    convert_slam_map_to_internal,
    generate_sample_map,
)
from .viz_utils import PathPublisher              # Publishes nav_msgs/Path for RViz
from .map_utils import MapPublisher               # Publishes an OccupancyGrid for RViz
from .utils import (
    is_cell_traversable,
    compute_obstacle_distance_map,
    mark_robot_footprint_clean,
    mark_footprint_along_path,
)

# TF2 to query robot pose (map -> base_link)
from tf2_ros import Buffer, TransformListener


class CleanerNode(Node):
    """
    A ROS 2 node that plans a path over a SLAM-produced occupancy grid and
    either:
      - sends it to Nav2's controller via FollowPath (real robot control), or
      - falls back to a local simulation (RViz-only visualization) if Nav2 is unavailable.

    Design goals:
      - Keep global planning self-contained (Python-side A*).
      - Delegate motion execution to Nav2's controller when present.
      - Preserve a simulation fallback so developers can test in RViz without Nav2.
    """

    def __init__(self) -> None:
        super().__init__('cleaner_node')

        # ===========================
        # Parameters / Robot geometry
        # ===========================
        # Robot radius in meters. Used for feasibility checks in grid space.
        # NOTE: Make sure this matches (approximately) Nav2's footprint/inflation parameters
        #       to avoid discrepancies between planned "can pass" and controller "cannot pass".
        self.robot_radius_m: float = 0.283

        # Default map resolution (m/cell). Will be overwritten by SLAM's actual resolution
        # once the /map message arrives. Kept as a sane default for internal-map fallback.
        self.map_resolution: float = 0.05
        self.robot_radius_cells: int = int(np.ceil(self.robot_radius_m / self.map_resolution))

        # =================
        # Internal state
        # =================
        # Internal grid map. See encoding in the module docstring.
        self.grid_map: np.ndarray | None = None
        # Optional precomputed distance/cost map for A* (same shape as grid_map).
        self.distance_map: np.ndarray | None = None
        # Flag set after first map is processed.
        self.map_ready: bool = False
        # Robot position in grid indices (row, col). Used as a fallback if TF is unavailable.
        self.robot_pos: list[int] = [10, 10]

        # For RViz visualization:
        # - 'path' stores (col, row) pairs for the Path publisher (x=col, y=row in grid space).
        self.path: list[tuple[int, int]] = []
        # 'cleaned' keeps the set of visited cells to avoid double counting.
        self.cleaned: set[tuple[int, int]] = set()
        # Simple performance/debug counter for steps taken in simulation.
        self.total_steps: int = 0

        # Map origin (x, y) in meters in the "map" frame.
        # OccupancyGrid's cell (0,0) is located at (origin.x, origin.y).
        # Rotation is assumed to be identity (w=1) in common setups; if your map origin
        # includes yaw, you must rotate (col*res, row*res) before adding the origin.
        self.map_origin_xy: tuple[float, float] = (0.0, 0.0)

        # =================
        # Publishers
        # =================
        self.path_pub = PathPublisher(self)   # nav_msgs/Path for RViz
        self.map_pub = MapPublisher(self)     # OccupancyGrid reflecting cleaned cells

        # =================
        # Subscriptions
        # =================
        # Subscribe to SLAM map. If your system uses a namespace, remap this in a launch file.
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10  # QoS depth; for production, consider aligning with map server QoS
        )

        # =================
        # Nav2 interface
        # =================
        # Action client that talks to Nav2's FollowPath server (controller_server).
        self.path_executor = PathExecutor(self)

        # =================
        # Simulation support
        # =================
        self.sim_path: list[tuple[int, int]] = []
        self.clean_timer = None  # will be a ROS timer when simulation is active

        # ==========
        # Parameters
        # ==========
        # Time to wait for a SLAM map before falling back to an internal test map.
        self.declare_parameter("slam_timeout", 5.0)
        self.slam_timeout: float = self.get_parameter("slam_timeout").get_parameter_value().double_value
        self.start_time: float = self.get_clock().now().nanoseconds / 1e9

        # ================
        # TF (robot pose)
        # ================
        # We fetch the robot pose as map->base_link to derive a good starting cell.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_timeout = Duration(seconds=0.5)

        # =========================
        # Housekeeping / Main timer
        # =========================
        # Poll until we get a map or until a timeout triggers the internal test map.
        self.timer = self.create_timer(0.05, self.wait_for_map_or_start)  # 20 Hz

        self.get_logger().info("🟢 Cleaner node initialized and waiting for map.")

    # ------------------------------------------------------------------------------
    # Map handling
    # ------------------------------------------------------------------------------
    def map_callback(self, msg: OccupancyGrid) -> None:
        """
        Handle the first SLAM /map message:
          1) Convert it into the internal grid representation (0/1/2).
          2) Build an optional distance map for A* heuristics.
          3) Cache resolution and origin for cell->world conversions.

        This callback runs only once (first map): subsequent /map updates are ignored
        to keep the logic simple. If you want to handle rolling map updates, you can
        extend this (e.g., rebuild distance map and re-plan).
        """
        if self.map_ready:
            return

        # Convert OccupancyGrid's flat array into the internal 2D grid (int)
        grid = convert_slam_map_to_internal(
            msg.data,
            msg.info.width,
            msg.info.height
        )
        self.grid_map = grid

        # Precompute distances to obstacles for planning (optional but useful).
        self.distance_map = compute_obstacle_distance_map(grid)

        # Use SLAM-provided resolution to keep meters <-> cells consistent.
        self.map_resolution = msg.info.resolution
        self.robot_radius_cells = int(np.ceil(self.robot_radius_m / self.map_resolution))

        # Cache map origin. If your map origin has a non-zero yaw, rotate cell coordinates
        # (col*res, row*res) accordingly before translating by (origin.x, origin.y).
        self.map_origin_xy = (
            msg.info.origin.position.x,
            msg.info.origin.position.y
        )

        self.map_ready = True
        self.get_logger().info("✅ SLAM map received. Starting cleaning.")
        self.start_cleaning()

    def wait_for_map_or_start(self) -> None:
        """
        If no SLAM map arrives within 'slam_timeout', construct an internal synthetic map
        so you can test planning and RViz visualization offline (e.g., when Nav2/SLAM is not running).
        """
        if self.map_ready:
            self.timer.cancel()
            return

        elapsed = self.get_clock().now().nanoseconds / 1e9 - self.start_time
        if elapsed > self.slam_timeout:
            self.get_logger().warn("⚠️ No SLAM map received in time. Using internal map.")
            self.grid_map = generate_sample_map()
            self.distance_map = compute_obstacle_distance_map(self.grid_map)
            self.map_ready = True
            self.start_cleaning()
            self.timer.cancel()

    # ------------------------------------------------------------------------------
    # Robot pose & path utilities
    # ------------------------------------------------------------------------------
    def get_robot_cell(self) -> tuple[int, int]:
        """
        Query TF for the robot pose in the 'map' frame and convert to grid indices (row, col).

        Returns:
            (row, col) in grid coordinates. Falls back to the last known self.robot_pos
            if TF lookup fails (e.g., no transform available yet).

        Notes:
            - This assumes TF tree includes map -> base_link.
            - If your robot uses a different base frame (e.g., base_footprint), update it here.
        """
        try:
            tf = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(), self.tf_timeout
            )
            x = tf.transform.translation.x
            y = tf.transform.translation.y

            ox, oy = self.map_origin_xy
            col = int((x - ox) / self.map_resolution)
            row = int((y - oy) / self.map_resolution)

            # Defensive clamping to valid indices
            row = max(0, min(row, self.grid_map.shape[0] - 1))
            col = max(0, min(col, self.grid_map.shape[1] - 1))
            return row, col

        except Exception as e:
            # Typical causes: TF not yet available, frames not published, etc.
            self.get_logger().warn(f"TF lookup failed, fallback to last pos: {e}")
            return self.robot_pos[0], self.robot_pos[1]

    def densify_cells(self, path_rc: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """
        Densify a grid path by inserting intermediate cells between neighbor pairs.
        This helps Nav2 controllers (e.g., RegulatedPurePursuit) to track more smoothly.

        Args:
            path_rc: list of (row, col) grid cells from A*.

        Returns:
            A denser list of (row, col) cells. If the input has <2 points, it's returned as-is.

        Implementation detail:
            We do simple linear interpolation in index space. For 4-connected A* paths, this
            effectively expands straight segments to per-cell steps and preserves corners.
        """
        if not path_rc or len(path_rc) < 2:
            return path_rc

        out: list[tuple[int, int]] = []
        for (r1, c1), (r2, c2) in zip(path_rc[:-1], path_rc[1:]):
            out.append((r1, c1))
            dr, dc = (r2 - r1), (c2 - c1)
            steps = max(abs(dr), abs(dc))
            # Fill intermediate cells using integer interpolation
            for k in range(1, steps):
                out.append((r1 + round(dr * k / steps), c1 + round(dc * k / steps)))
        out.append(path_rc[-1])
        return out

    # ------------------------------------------------------------------------------
    # Main planning & execution entry
    # ------------------------------------------------------------------------------
    def start_cleaning(self) -> None:
        """
        Publish the current internal map (for RViz), compute a goal cell,
        plan a path (A*), and either:
          - send it to Nav2's FollowPath (real robot execution), or
          - fall back to local simulation if FollowPath is unavailable.

        Replanning:
          - In simulation mode, when sliding stops or target is reached, we re-run BFS+A*.
          - In Nav2 mode, if you need online replanning (e.g., dynamic obstacles),
            you can trigger a new A* here (or on a timer) and call send_path again.
        """
        # Ensure RViz sees the current map state
        self.map_pub.publish_map(self.grid_map)

        # Start position from TF (falls back to last-known if TF isn't ready)
        r, c = self.get_robot_cell()
        self.robot_pos = [r, c]

        # Nothing to do if there are no unclean cells (grid==1)
        if not np.any(self.grid_map == 1):
            self.get_logger().info("✅ No cleaning needed. Map already clean.")
            return

        # Pick a target cell to clean next: nearest unclean reachable (BFS)
        goal = bfs_nearest_unclean(
            self.grid_map, r, c,
            robot_radius_cells=self.robot_radius_cells
        )
        if goal is None:
            self.get_logger().warn("⚠️ No reachable unclean cell found.")
            return

        # Global path planning (A*). Uses distance_map to shape cost if available.
        path = astar(
            self.grid_map, (r, c), goal,
            robot_radius_cells=self.robot_radius_cells,
            distance_map=self.distance_map,
            alpha=1.0
        )
        if not path:
            self.get_logger().warn("⚠️ A* failed to find a path.")
            return

        # Densify to help controller tracking (keeps same geometry, finer sampling).
        path = self.densify_cells(path)

        # Convert grid cells (row, col) to world coordinates (x, y) using origin + resolution.
        ox, oy = self.map_origin_xy
        path_in_meters = [
            (ox + col * self.map_resolution, oy + row * self.map_resolution)
            for row, col in path
        ]

        # Try Nav2 controller. If unavailable (e.g., controller_server not started),
        # switch to local simulation so you can still see progress in RViz.
        if not self.path_executor.client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("❌ FollowPath server not available. Falling back to simulation.")
            self.sim_path = path
            self.start_cleaning_simulation()
            return

        # Nav2 is available -> send path to controller for execution on the robot
        self.path_executor.send_path(path_in_meters)

    # ------------------------------------------------------------------------------
    # Local simulation fallback (RViz-only)
    # ------------------------------------------------------------------------------
    def start_cleaning_simulation(self) -> None:
        """
        Start a local cleaning simulation loop (20 Hz). This updates the internal map and
        publishes the visualized path so you can watch the cleaning behavior in RViz
        without any Nav2/real robot.
        """
        self.get_logger().warn("⚠️ Running local simulation instead of real robot control.")
        self.clean_timer = self.create_timer(0.05, self.clean_step)  # 20 Hz

    def clean_step(self) -> None:
        """
        One simulation step (called at 20 Hz):
          - Mark the current footprint as cleaned.
          - Attempt to slide the robot forward by one 'footprint step' in a 4-neighborhood.
          - If cannot slide, replan to the nearest unclean target and step along that path.
          - Continuously publish the path & map for RViz.
        """
        r, c = self.robot_pos

        # Mark current robot footprint as cleaned
        mark_robot_footprint_clean(self.grid_map, r, c, self.robot_radius_cells)
        self.cleaned.add((r, c))
        self.total_steps += 1

        # Simple coverage-style sliding: try 4 directions (R, U, L, D) by one footprint step.
        step = max(1, self.robot_radius_cells)
        moved = False
        for dr, dc in [(0, 1), (-1, 0), (0, -1), (1, 0)]:
            nr, nc = r + dr * step, c + dc * step
            if (self.is_valid(nr, nc)
                and self.grid_map[nr, nc] == 1
                and is_cell_traversable(self.grid_map, nr, nc, self.robot_radius_cells)):
                self.robot_pos = [nr, nc]
                moved = True
                break

        # If sliding fails, replan to the nearest unclean spot and walk along that path.
        if not moved:
            if not np.any(self.grid_map == 1):
                self.get_logger().info(f"✅ Simulation cleaning complete! Steps: {self.total_steps}")
                self.destroy_timer(self.clean_timer)
                return

            goal = bfs_nearest_unclean(
                self.grid_map, r, c,
                robot_radius_cells=self.robot_radius_cells
            )
            if goal is None:
                self.get_logger().warn("⚠️ No reachable unclean cell in simulation.")
                self.destroy_timer(self.clean_timer)
                return

            path = astar(
                self.grid_map, (r, c), goal,
                robot_radius_cells=self.robot_radius_cells,
                distance_map=self.distance_map,
                alpha=1.0
            )

            # Step along the planned path, marking footprint at each cell.
            if path:
                for prev, curr in zip(path[:-1], path[1:]):
                    self.robot_pos = list(curr)
                    r, c = self.robot_pos
                    mark_footprint_along_path(self.grid_map, prev, curr, self.robot_radius_cells)
                    self.cleaned.add((r, c))
                    self.total_steps += 1

                    # For RViz path visualization we store (col, row) pairs.
                    self.path.append((c, r))
                    self.path_pub.publish_path(self.path)
                    self.map_pub.publish_map(self.grid_map)
                    time.sleep(0.01)  # tiny sleep for visual smoothness (simulation only)

        # Always append/publish the latest simulated pose for RViz
        self.path.append((c, r))
        self.path_pub.publish_path(self.path)
        self.map_pub.publish_map(self.grid_map)

    # ------------------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------------------
    def is_valid(self, r: int, c: int) -> bool:
        """
        Check if (row, col) is inside the current grid bounds.
        NOTE: Only safe to call after self.grid_map is initialized.
        """
        return 0 <= r < self.grid_map.shape[0] and 0 <= c < self.grid_map.shape[1]


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main(args=None) -> None:
    rclpy.init(args=args)
    node = CleanerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

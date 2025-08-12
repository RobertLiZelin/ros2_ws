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
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from nav_msgs.msg import OccupancyGrid

# Project modules
from .path_executor import PathExecutor
from .astar import astar
from .bfs import bfs_nearest_unclean
from .map_manager import (
    convert_slam_map_to_internal,
    generate_sample_map,
)
from .viz_utils import PathPublisher
from .map_utils import MapPublisher
from .utils import (
    is_cell_traversable,
    compute_obstacle_distance_map,
    mark_robot_footprint_clean,
    mark_footprint_along_path,
)

# TF2 for robot pose lookup (map -> base_link)
from tf2_ros import Buffer, TransformListener


class CleanerNode(Node):
    """
    A ROS 2 node that:
    - Receives an OccupancyGrid from SLAM
    - Plans a global path using A* / BFS
    - Executes via Nav2 or falls back to internal simulation for visualization
    """

    def __init__(self) -> None:
        super().__init__('cleaner_node')

        # ===========================
        # Robot geometry parameters
        # ===========================
        self.robot_radius_m: float = 0.283  # Robot radius in meters
        self.map_resolution: float = 0.05   # Default resolution (m/cell), updated by SLAM map
        self.robot_radius_cells: int = int(np.ceil(self.robot_radius_m / self.map_resolution))

        # =================
        # Internal state
        # =================
        self.grid_map: np.ndarray | None = None
        self.distance_map: np.ndarray | None = None
        self.map_ready: bool = False
        self.robot_pos: list[int] = [10, 10]
        self.path: list[tuple[int, int]] = []
        self.cleaned: set[tuple[int, int]] = set()
        self.total_steps: int = 0
        self.map_origin_xy: tuple[float, float] = (0.0, 0.0)

        # =================
        # Parameters
        # =================
        # Timeout to wait for SLAM map before switching to an internal test map (extended to 15s)
        self.declare_parameter("slam_timeout", 15.0)
        self.slam_timeout: float = self.get_parameter("slam_timeout").get_parameter_value().double_value

        # Configurable map topic and frame
        self.map_topic: str = self.declare_parameter("map_topic", "/map").value
        self.map_frame: str = self.declare_parameter("map_frame", "map").value

        self.start_time: float = self.get_clock().now().nanoseconds / 1e9

        # =================
        # Publishers
        # =================
        self.path_pub = PathPublisher(self)
        self.map_pub = MapPublisher(self)

        # =================
        # Subscriptions
        # =================
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,  # Important for latched map
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            map_qos
        )

        # =================
        # Nav2 interface
        # =================
        self.path_executor = PathExecutor(self)

        # ================
        # TF setup
        # ================
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_timeout = Duration(seconds=0.5)

        # =========================
        # Main timer
        # =========================
        self.timer = self.create_timer(0.05, self.wait_for_map_or_start)

        self.get_logger().info(f"🟢 Cleaner node initialized. Waiting for map on '{self.map_topic}' with TRANSIENT_LOCAL QoS.")

    # ------------------------------------------------------------------------------
    # Map handling
    # ------------------------------------------------------------------------------
    def map_callback(self, msg: OccupancyGrid) -> None:
        """Handle the first SLAM map message and convert to internal grid format."""
        if self.map_ready:
            return

        self.grid_map = convert_slam_map_to_internal(
            msg.data,
            msg.info.width,
            msg.info.height
        )
        self.distance_map = compute_obstacle_distance_map(self.grid_map)
        self.map_resolution = msg.info.resolution
        self.robot_radius_cells = int(np.ceil(self.robot_radius_m / self.map_resolution))
        self.map_origin_xy = (
            msg.info.origin.position.x,
            msg.info.origin.position.y
        )
        self.map_ready = True
        self.get_logger().info("✅ SLAM map received. Starting cleaning.")
        self.start_cleaning()

    def wait_for_map_or_start(self) -> None:
        """Wait for SLAM map, or fall back to internal test map after timeout."""
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
        """Get robot position in grid coordinates from TF."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, 'base_link', rclpy.time.Time(), self.tf_timeout
            )
            x = tf.transform.translation.x
            y = tf.transform.translation.y
            ox, oy = self.map_origin_xy
            col = int((x - ox) / self.map_resolution)
            row = int((y - oy) / self.map_resolution)
            row = max(0, min(row, self.grid_map.shape[0] - 1))
            col = max(0, min(col, self.grid_map.shape[1] - 1))
            return row, col
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed, fallback to last pos: {e}")
            return self.robot_pos[0], self.robot_pos[1]

    def densify_cells(self, path_rc: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Insert intermediate cells between path points for smoother controller tracking."""
        if not path_rc or len(path_rc) < 2:
            return path_rc
        out: list[tuple[int, int]] = []
        for (r1, c1), (r2, c2) in zip(path_rc[:-1], path_rc[1:]):
            out.append((r1, c1))
            dr, dc = (r2 - r1), (c2 - c1)
            steps = max(abs(dr), abs(dc))
            for k in range(1, steps):
                out.append((r1 + round(dr * k / steps), c1 + round(dc * k / steps)))
        out.append(path_rc[-1])
        return out

    # ------------------------------------------------------------------------------
    # Main cleaning logic
    # ------------------------------------------------------------------------------
    def start_cleaning(self) -> None:
        """Publish current map, find target cell, plan with A*, and execute."""
        self.map_pub.publish_map(self.grid_map)
        r, c = self.get_robot_cell()
        self.robot_pos = [r, c]

        if not np.any(self.grid_map == 1):
            self.get_logger().info("✅ No cleaning needed. Map already clean.")
            return

        goal = bfs_nearest_unclean(
            self.grid_map, r, c, robot_radius_cells=self.robot_radius_cells
        )
        if goal is None:
            self.get_logger().warn("⚠️ No reachable unclean cell found.")
            return

        path = astar(
            self.grid_map, (r, c), goal,
            robot_radius_cells=self.robot_radius_cells,
            distance_map=self.distance_map,
            alpha=1.0
        )
        if not path:
            self.get_logger().warn("⚠️ A* failed to find a path.")
            return

        path = self.densify_cells(path)
        ox, oy = self.map_origin_xy
        path_in_meters = [
            (ox + col * self.map_resolution, oy + row * self.map_resolution)
            for row, col in path
        ]

        if not self.path_executor.client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("❌ FollowPath server not available. Falling back to simulation.")
            self.sim_path = path
            self.start_cleaning_simulation()
            return

        self.path_executor.send_path(path_in_meters)

    # ------------------------------------------------------------------------------
    # Local simulation
    # ------------------------------------------------------------------------------
    def start_cleaning_simulation(self) -> None:
        """Simulate cleaning in RViz without controlling a real robot."""
        self.get_logger().warn("⚠️ Running local simulation instead of real robot control.")
        self.clean_timer = self.create_timer(0.05, self.clean_step)

    def clean_step(self) -> None:
        """One step of cleaning simulation."""
        r, c = self.robot_pos
        mark_robot_footprint_clean(self.grid_map, r, c, self.robot_radius_cells)
        self.cleaned.add((r, c))
        self.total_steps += 1
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

        if not moved:
            if not np.any(self.grid_map == 1):
                self.get_logger().info(f"✅ Simulation cleaning complete! Steps: {self.total_steps}")
                self.destroy_timer(self.clean_timer)
                return
            goal = bfs_nearest_unclean(
                self.grid_map, r, c, robot_radius_cells=self.robot_radius_cells
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
            if path:
                for prev, curr in zip(path[:-1], path[1:]):
                    self.robot_pos = list(curr)
                    r, c = self.robot_pos
                    mark_footprint_along_path(self.grid_map, prev, curr, self.robot_radius_cells)
                    self.cleaned.add((r, c))
                    self.total_steps += 1
                    self.path.append((c, r))
                    self.path_pub.publish_path(self.path)
                    self.map_pub.publish_map(self.grid_map)
                    time.sleep(0.01)
        self.path.append((c, r))
        self.path_pub.publish_path(self.path)
        self.map_pub.publish_map(self.grid_map)

    # ------------------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------------------
    def is_valid(self, r: int, c: int) -> bool:
        """Check if coordinates are within grid bounds."""
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

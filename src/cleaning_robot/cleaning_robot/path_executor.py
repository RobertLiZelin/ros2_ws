from rclpy.action import ActionClient
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import FollowPath


class PathExecutor:
    """
    Handles sending pre-computed paths to the Nav2 `FollowPath` action server.

    This class acts as a bridge between your custom path planner
    and Nav2's execution engine.
    """

    def __init__(self, node):
        """
        Initialize the PathExecutor.

        Args:
            node (rclpy.node.Node): ROS2 node instance.
        """
        self.node = node
        # Create an ActionClient to communicate with Nav2's FollowPath server
        self.client = ActionClient(node, FollowPath, 'follow_path')

    def convert_path_to_nav_path(self, path_points, frame_id="map"):
        """
        Convert a list of (x, y) coordinates into a ROS2 Path message.

        Args:
            path_points (list of tuple): List of (x, y) positions in meters.
            frame_id (str): Reference frame for the path (default: "map").

        Returns:
            Path: ROS2 Path message containing all poses.
        """
        path_msg = Path()
        path_msg.header.frame_id = frame_id
        path_msg.header.stamp = self.node.get_clock().now().to_msg()

        for x, y in path_points:
            pose = PoseStamped()
            pose.header = path_msg.header  # Same timestamp/frame for all poses
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0  # Facing forward (no rotation applied)
            path_msg.poses.append(pose)

        return path_msg

    def send_path(self, path_points):
        """
        Send the computed path to the Nav2 FollowPath action server.

        If the Nav2 controller is unavailable, logs an error.

        Args:
            path_points (list of tuple): List of (x, y) positions in meters.
        """
        # Check if FollowPath server is available
        if not self.client.wait_for_server(timeout_sec=5.0):
            self.node.get_logger().error("❌ FollowPath server not available")
            return

        # Convert (x, y) list to Path message
        path_msg = self.convert_path_to_nav_path(path_points)

        # Create a FollowPath goal and assign the path
        goal = FollowPath.Goal()
        goal.path = path_msg

        # Log and send asynchronously
        self.node.get_logger().info("📤 Sending path to Nav2 controller")
        self.client.send_goal_async(goal)

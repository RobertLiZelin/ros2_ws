import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


class PathPublisher:
    """
    Publishes a `nav_msgs/Path` message to visualize the robot's traversed path in RViz.

    This utility class allows any ROS 2 node to publish a sequence of positions
    (x, y in meters) as a continuous path. Each point in the path is wrapped
    in a `geometry_msgs/PoseStamped` with a fixed orientation (no rotation).

    Attributes:
        node (Node):
            The ROS 2 node instance using this publisher. Needed to create publishers
            and access the clock.
        publisher (Publisher):
            ROS 2 publisher for the `/cleaned_path` topic.
        frame_id (str):
            Coordinate frame in which the path is expressed (default: "map").
    """

    def __init__(self, node: Node):
        """
        Initialize the PathPublisher.

        Args:
            node (Node): The parent ROS 2 node that owns this publisher.
        """
        self.node = node
        # Create publisher for `nav_msgs/Path` on topic `/cleaned_path`
        self.publisher = node.create_publisher(Path, '/cleaned_path', 10)
        # Frame of reference for the path (e.g., 'map' from SLAM)
        self.frame_id = 'map'

    def publish_path(self, path_points):
        """
        Publish a sequence of (x, y) points as a `nav_msgs/Path`.

        Args:
            path_points (list[tuple[float, float]]):
                A list of 2D coordinates (in meters) representing the robot's path.
                Coordinates are given in (x, y) format in the `frame_id` reference frame.

        Notes:
            - The `z` coordinate of all points is set to 0.0 (2D planar assumption).
            - The orientation is fixed with `w=1.0` (identity quaternion: no rotation).
            - All poses share the same timestamp for simplicity, matching the header stamp.
        """
        msg = Path()
        # Timestamp the path message with the current ROS time
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        for x, y in path_points:
            pose = PoseStamped()
            pose.header.stamp = msg.header.stamp
            pose.header.frame_id = self.frame_id
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0  # Identity quaternion: no rotation
            msg.poses.append(pose)

        # Publish the constructed Path message
        self.publisher.publish(msg)

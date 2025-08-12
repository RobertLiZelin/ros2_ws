from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
import numpy as np


class MapPublisher:
    """
    Publishes the internal cleaning map to a ROS 2 topic (`/cleaning_map`) 
    in the `nav_msgs/OccupancyGrid` format for visualization (e.g., in RViz).

    This allows other ROS nodes and visualization tools to see:
    - Obstacles
    - Uncleaned areas
    - Cleaned areas
    """

    def __init__(self, node):
        """
        Initialize the MapPublisher.

        Args:
            node (rclpy.node.Node):
                The ROS 2 node that will own this publisher.
        """
        self.node = node
        # Create a publisher for OccupancyGrid messages on the '/cleaning_map' topic
        self.pub = node.create_publisher(OccupancyGrid, '/cleaning_map', 10)

    def publish_map(self, grid_map):
        """
        Convert the internal cleaning map into an OccupancyGrid and publish it.

        Args:
            grid_map (np.ndarray):
                2D array representing the cleaning map:
                - `0` = obstacle
                - `1` = uncleaned free space
                - `2` = cleaned space
                - other values = unknown
        """
        msg = OccupancyGrid()

        # ======== Header setup ========
        msg.header = Header()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = 'map'  # Reference frame for the map

        # ======== Map metadata ========
        # NOTE: Resolution here is currently set to 1.0 m per cell for simplicity.
        #       In a real robot, you should match this to the SLAM map resolution.
        msg.info.resolution = 1.0  
        msg.info.width = grid_map.shape[1]   # Number of columns
        msg.info.height = grid_map.shape[0]  # Number of rows

        # Map origin in world coordinates (bottom-left corner of the map)
        msg.info.origin.position.x = 0.0
        msg.info.origin.position.y = 0.0
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0  # No rotation

        # ======== Convert internal map to OccupancyGrid.data ========
        data = []
        for r in range(grid_map.shape[0]):
            for c in range(grid_map.shape[1]):
                val = grid_map[r, c]
                if val == 0:
                    data.append(100)   # Obstacle (100 = fully occupied)
                elif val == 1:
                    data.append(0)     # Free, uncleaned cell
                elif val == 2:
                    data.append(50)    # Cleaned cell (partial occupancy for visualization)
                else:
                    data.append(-1)    # Unknown space

        msg.data = data

        # Publish the message
        self.pub.publish(msg)

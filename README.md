# ROS 2 Lawnbot

A ROS 2 Jazzy-based coverage and weeding demo robot. This README provides a quick start, VS Code workflow, SLAM/Nav2 hooks, visualization topics, and troubleshooting. It’s adapted from your internal tutorial PDF and organized for GitHub.

> **Tested on**: Ubuntu 24.04 (WSL2) + ROS 2 Jazzy

---

## 1) Quick Start (build & run)

```bash
# From your ROS 2 workspace root
cd ~/ros2_ws

# Build just this package
colcon build --packages-select cleaning_robot

# Source the workspace
source install/setup.bash

# Run the node
ros2 run cleaning_robot cleaner_node
```

If you need system ROS environment in a fresh terminal:
```bash
source /opt/ros/jazzy/setup.bash
```

---

## 2) VS Code workflow

1. Install extensions: **Remote - WSL** and **ROS** (ms-iot.vscode-ros).
2. Open the workspace:
   ```bash
   code ~/ros2_ws
   ```
3. Use the built-in terminal and ensure your environment is sourced:
   ```bash
   source /opt/ros/jazzy/setup.bash
   ```
4. Build & run from the terminal as shown in **Quick Start**.

---

## 3) SLAM / Map integration

- `cleaner_node` subscribes to `/map` (`nav_msgs/OccupancyGrid`).  
- If a SLAM map is available, it is used directly; otherwise the node falls back to a simulated map.
- The map is converted to an internal grid for planning without resampling.

---

## 4) Navigation (Nav2) hook

- The node publishes planned coverage paths to `/cleaned_path` (`nav_msgs/Path`).  
- To drive a controller, subscribe to `/cleaned_path` and forward to your Nav2 action server / controller plugin.

---

## 5) Visualization

Visualize these topics in **RViz2**:
- `/cleaned_path` — planned path (`nav_msgs/Path`)
- `/cleaning_map` — visualization map (`nav_msgs/OccupancyGrid`)

---

## 6) Repository structure (key files)

```
cleaning_robot/
 ├─ cleaning_robot/
 │   ├─ cleaner_node.py     # main orchestration
 │   ├─ astar.py            # A* path planner
 │   ├─ bfs.py              # BFS coverage support
 │   ├─ map_manager.py      # map conversion & generation
 │   ├─ map_utils.py        # publish /cleaning_map
 │   └─ viz_utils.py        # publish /cleaned_path
 └─ package.xml / setup.cfg / setup.py
```

> *File names may vary slightly depending on your local repo; adjust accordingly.*

---

## 7) Troubleshooting

- **colcon permission or cache issues**
  ```bash
  sudo rm -rf install/ build/ log/
  colcon build --packages-select cleaning_robot
  ```

- **RViz/VNC issues on Raspberry Pi**
  Ensure the desktop session is active and use **x11vnc** or **tightvncserver**.

---

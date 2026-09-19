# Mobile Manipulation & Robotics Simulation

An end-to-end 2D mobile manipulation pipeline implemented in Python. The system simulates autonomous navigation, real-time perception, local control optimization, online learning, and arm manipulation to accomplish a pick-and-place task in a cluttered environment.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture & Subsystems](#architecture--subsystems)
   - [World Representation & Grid Inflation](#1-world-representation--grid-inflation)
   - [Global Motion Planning](#2-global-motion-planning)
   - [Perception & Computer Vision](#3-perception--computer-vision)
   - [Kinematics & Manipulation](#4-kinematics--manipulation)
   - [Local Control & Optimization](#5-local-control--optimization)
   - [Online Behavior Cloning](#6-online-behavior-cloning)
3. [Mission Execution Flow](#mission-execution-flow)
4. [File Breakdown](#file-breakdown)
5. [Getting Started](#getting-started)

---

## Project Overview

This project models a differential-drive robot equipped with a 2-DOF articulated arm, a planar LiDAR sensor, and an ego-centric visual camera. The goal of the system is to autonomously navigate from a start position $(2.0, 2.0)$ across a $40 \times 30$ meter obstacle course to a target object located at $(37.0, 27.0)$, position its manipulator arm, and grasp the object.

---

## Architecture & Subsystems

The codebase is split into core algorithmic building blocks (`robot_core.py`) and a mission state machine with visual reporting (`robot_sim.py`).

### 1. World Representation & Grid Inflation
- **Occupancy Grid**: The environment is represented as a binary occupancy matrix derived from world dimensions and spatial resolution.
- **Safety Margin Expansion (`_inflate`)**: To prevent collisions due to the robot's physical dimensions, static obstacles are expanded using morphological binary dilation (`np.pad` and neighbor OR operations) before motion planning occurs.

### 2. Global Motion Planning
The system implements two distinct global path planning algorithms:
- **$A^*$ Planning (`astar`)**: Evaluates a 2D grid with 8-connectivity (diagonal moves cost $\sqrt{2}$, cardinal moves cost $1.0$). It uses Euclidean distance as an admissible heuristic to compute the optimal path from start to target.
- **Rapidly-exploring Random Trees (`rrt`)**: A sampling-based planner that iteratively builds a tree across the continuous workspace using random sampling with goal-biasing ($15\%$ probability). It provides an alternative trajectory comparison against $A^*$.
- **Segment Checking (`_seg_free`)**: Validates line-of-sight clearance along path segments using linear interpolation.

### 3. Perception & Computer Vision
- **Ray-Casting LiDAR (`lidar_scan`)**: Simulates 32 radial laser rays across a $360^\circ$ field of view up to a max range of 7 meters. It computes ray-grid intersections to return hit coordinates and minimum obstacle clearances.
- **Ego Camera Simulation (`camera_image`)**: Projects a forward-facing local subgrid relative to the robot's heading $\theta$ to render an image of the visible local environment.
- **Edge Detection (`sobel_edges`)**: Applies a 2D spatial Sobel filter along horizontal ($G_x$) and vertical ($G_y$) axes to compute normalized edge gradient magnitudes:
  $$M = \sqrt{G_x^2 + G_y^2}$$

### 4. Kinematics & Manipulation
- **Mobile Base Step (`Robot.step`)**: Implements unicycle drive dynamics governed by linear velocity $v$ and angular velocity $\omega$. Includes collision check guards before committing position updates.
- **Inverse Kinematics (`Robot.arm_ik`)**: Solves closed-form IK for a 2-joint planar arm with link lengths $L_1 = 1.2\text{m}$ and $L_2 = 1.0\text{m}$. Transforms target coordinates into base-local space and calculates joint angles $(q_1, q_2)$ using the Law of Cosines:
  $$\cos(q_2) = \frac{d^2 - L_1^2 - L_2^2}{2 L_1 L_2}$$
- **Forward Kinematics (`Robot.arm_points`)**: Computes global joint positions $(p_1, p_2)$ for drawing and tip-distance validation during grasping.

### 5. Local Control & Optimization
- **Path Follower (`follow_path`)**: A sequential waypoint tracker that dynamically advances target index based on lookahead distance and direct line-of-sight segment checks. Adjusts linear speed based on heading error to smooth turns.
- **Model Predictive Control (`mpc`)**: Activated when proximity to obstacles falls below a critical threshold ($< 1.5\text{m}$). Evaluates trajectory candidates over a horizon $H=6$ using random-shooting optimization to minimize goal distance, turning effort, and collision risk.
- **Stuck Recovery**: Monitors robot displacement over a 30-step sliding window. If position drift is less than $0.5\text{m}$, an unstuck override sequence triggers a reverse-and-turn maneuver.

### 6. Online Behavior Cloning
- **Least Mean Squares Regressor (`OnlineLinear`)**: Performs online supervised learning to clone the behavior of the global path follower. 
- **Feature Vector**: Takes normalized relative goal distance, relative heading angle $(\sin\theta, \cos\theta)$, and minimum LiDAR clearance as input to predict control commands $(v, \omega)$ on each time step.

---

## Mission Execution Flow

The simulation runs a state machine with four operational phases:

```
┌──────┐    Goal Dist < 1.5m    ┌───────┐    Arm Tip Error < 0.2m    ┌───────┐    Timer > 20 steps    ┌──────┐
│ NAV  │ ─────────────────────> │ REACH │ ─────────────────────────> │ GRASP │ ─────────────────────> │ DONE │
└──────┘                        └───────┘                            └───────┘                         └──────┘
```

1. **NAV (Navigation)**: Follows the planned $A^*$ path. Dynamically switches between Pure Pursuit, MPC (for obstacle avoidance), and Unstuck recovery modes while concurrently training the online LMS regressor.
2. **REACH (Arm Alignment)**: Stops the mobile base upon reaching the goal vicinity. Drives arm joints $(q_1, q_2)$ towards the IK solution using proportional smooth interpolation.
3. **GRASP (Object Manipulation)**: Actuates the gripper mechanism until maximum closure is achieved.
4. **DONE (Mission Complete)**: Halts execution, outputs telemetry stats, and exports visualization plots.

---

## File Breakdown

| File | Purpose |
| :--- | :--- |
| **`robot_core.py`** | Modular library containing algorithm implementations (`World`, $A^*$, RRT, LiDAR, CV, `Robot` model, IK, MPC, and LMS regressor). |
| **`robot_sim.py`** | Main entry point running the state machine loop, collecting telemetry metrics, and plotting `robot_report.png`. |

---

## Getting Started

### Prerequisites
Make sure you have Python 3.x along with the required libraries installed:
```bash
pip install numpy matplotlib
```

### Running the Simulation
Execute the main simulation script:
```bash
python robot_sim.py
```

### Generated Outputs
Upon completion, the terminal displays full mission telemetry (runtime, final heading, controller usage breakdown, ML convergence error, and arm state). It also generates **`robot_report.png`**, a 4-panel dashboard featuring:
- **World Map**: Global $A^*$ vs. RRT paths, executed base trajectory, LiDAR hits, and robot arm configuration.
- **Ego Camera View**: Raw camera capture from the robot's perspective.
- **Sobel Filter Output**: Edge detection feature map.
- **Telemetry Charts**: History of linear speed ($v$), angular velocity ($\omega$), and ML learning error curve.

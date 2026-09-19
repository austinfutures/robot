"""
robot_sim.py - Headless mission + summary figure (Telemetry & Path only).
Run:  python robot_sim.py
"""
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

# Removed camera_image and sobel_edges imports
from robot_core import (World, astar, rrt, lidar_scan, Robot, 
                        follow_path, mpc, OnlineLinear)

NAV, REACH, GRASP, DONE = 0, 1, 2, 3


def main():
    np.random.seed(0)
    t_start = time.time()

    world = World()
    start, goal = (2.0, 2.0), (37.0, 27.0)
    obj_pos = np.array(goal)

    print("[planning] A* ...")
    t0 = time.time(); path = astar(world, start, goal)
    print(f"            -> {len(path)} waypoints in {time.time()-t0:.3f}s")

    print("[planning] RRT ...")
    t0 = time.time(); rrt_path = rrt(world, start, goal)
    print(f"            -> {len(rrt_path) if rrt_path else 0} waypoints "
          f"in {time.time()-t0:.3f}s")

    robot = Robot(*start)
    ml = OnlineLinear(4, 2)
    state, timer = NAV, 0
    unstuck_timer = 0
    last_pos = (robot.x, robot.y)
    last_check = 0

    dt, max_steps = 0.1, 1500
    tx, ty = [], []
    hv, hw, he = [], [], []
    modes = {"follow": 0, "MPC": 0, "unstuck": 0}

    print("[sim] Running mission ...")
    t0 = time.time()
    steps, dg = 0, 999.0
    while steps < max_steps:
        ranges, hits = lidar_scan(world, robot.x, robot.y)
        min_r = float(ranges.min())

        if state == NAV:
            v_pp, w_pp, target_idx = follow_path(robot, path, world)
            v_cmd, w_cmd, mode = v_pp, w_pp, "follow"

            if unstuck_timer > 0:
                v_cmd, w_cmd = -0.8, robot.max_w * 0.6
                mode = "unstuck"
                unstuck_timer -= 1
            elif min_r < 1.5:
                tgt = path[min(target_idx + 2, len(path)-1)]
                v_cmd, w_cmd = mpc(world, robot, tgt)
                mode = "MPC"

            modes[mode] += 1
            robot.step(v_cmd, w_cmd, dt, world)

            # ML learns the pure-pursuit expert online
            dxg, dyg = goal[0]-robot.x, goal[1]-robot.y
            dg = float(np.hypot(dxg, dyg))
            ag = (np.arctan2(dyg, dxg) - robot.theta + np.pi) % (2*np.pi) - np.pi
            ml.update(np.array([dg/40, np.sin(ag), np.cos(ag), min_r/8]),
                      np.array([v_pp, w_pp]))

            # stuck detection every 30 steps
            if steps - last_check >= 30:
                moved = np.hypot(robot.x-last_pos[0], robot.y-last_pos[1])
                if moved < 0.5 and unstuck_timer == 0:
                    unstuck_timer = 20
                last_pos = (robot.x, robot.y)
                last_check = steps

            if dg < 1.5:
                state = REACH

        elif state == REACH:
            robot.step(0, 0, dt, world)
            tq1, tq2 = robot.arm_ik(obj_pos)
            robot.q1 += 0.20*(tq1-robot.q1); robot.q2 += 0.20*(tq2-robot.q2)
            tip = robot.arm_points()[-1]
            if np.hypot(tip[0]-obj_pos[0], tip[1]-obj_pos[1]) < 0.20:
                state, timer = GRASP, 0
            v_cmd = w_cmd = 0.0

        elif state == GRASP:
            robot.gripper = min(1.0, robot.gripper + 0.08)
            timer += 1
            if timer > 20:
                robot.has_object, state = True, DONE
            v_cmd = w_cmd = 0.0

        tx.append(robot.x); ty.append(robot.y)
        hv.append(v_cmd); hw.append(w_cmd); he.append(ml.last_err)
        steps += 1
        if state == DONE: break

    sim_time = time.time() - t0
    dg = float(np.hypot(goal[0]-robot.x, goal[1]-robot.y))
    print(f"[sim] Complete in {steps} steps ({sim_time:.2f}s)")

    print("\n===== TELEMETRY =====")
    print(f"  Final position   : ({robot.x:.2f}, {robot.y:.2f})")
    print(f"  Final heading    : {np.degrees(robot.theta):.1f} deg")
    print(f"  Distance to goal : {dg:.3f} m")
    print(f"  Controller usage : follow={modes['follow']} "
          f"MPC={modes['MPC']} unstuck={modes['unstuck']}")
    print(f"  ML final error   : {ml.last_err:.4f}")
    print(f"  Arm joint angles : q1={np.degrees(robot.q1):.1f}, "
          f"q2={np.degrees(robot.q2):.1f} deg")
    print(f"  Gripper          : {robot.gripper:.2f}")
    print(f"  Object grasped   : {robot.has_object}")
    print(f"  Mission status   : {'SUCCESS' if robot.has_object else 'INCOMPLETE'}")
    print(f"  Total runtime    : {time.time()-t_start:.2f}s")

    # ---------------- REPORT FIGURE (Path & Telemetry Only) ----------------
    fig, (ax_w, ax_t) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#0b0f14')
    for ax in (ax_w, ax_t):
        ax.set_facecolor('#101418')
        for s in ax.spines.values(): s.set_color('#445566')

    # --- Plot 1: World & Path ---
    ax_w.set_xlim(0, world.width); ax_w.set_ylim(0, world.height)
    ax_w.set_aspect('equal')
    ax_w.imshow(world.grid, origin='lower', cmap='Greys',
                extent=[0, world.width, 0, world.height], alpha=0.6)
    if path:
        px, py = zip(*path)
        ax_w.plot(px, py, '--', color='cyan', lw=1.5, label='A* path')
    if rrt_path:
        rx, ry = zip(*rrt_path)
        ax_w.plot(rx, ry, '-', color='orange', lw=1.0, alpha=0.7, label='RRT path')
    ax_w.plot(tx, ty, '-', color='lime', lw=2, label='trajectory')

    obj_color = 'lime' if robot.has_object else 'magenta'
    obj_xy = (obj_pos[0]-0.5, obj_pos[1]-0.5)
    if robot.has_object:
        tip = robot.arm_points()[-1]
        obj_xy = (tip[0]-0.5, tip[1]-0.5)
    ax_w.add_patch(Rectangle(obj_xy, 1, 1, color=obj_color, zorder=4))
    ax_w.add_patch(Circle((robot.x, robot.y), 0.6, color='orange', zorder=5))
    ax_w.plot([robot.x, robot.x + np.cos(robot.theta)],
              [robot.y, robot.y + np.sin(robot.theta)],
              '-', color='yellow', lw=2, zorder=6)
    pts = robot.arm_points()
    ax_w.plot(pts[:, 0], pts[:, 1], '-', color='white', lw=3, zorder=7)
    _, hits_final = lidar_scan(world, robot.x, robot.y)
    if len(hits_final):
        ax_w.plot(hits_final[:, 0], hits_final[:, 1], '.',
                  color='red', ms=3, alpha=0.6)
    ax_w.legend(loc='upper left', fontsize=8, facecolor='#0b0f14',
                edgecolor='#445566', labelcolor='white')
    ax_w.set_title("World: A* + RRT + follow_path + MPC",
                   color='white', fontsize=10)
    ax_w.set_xticks([]); ax_w.set_yticks([])

    # --- Plot 2: Telemetry ---
    ax_t.plot(hv, '-', color='cyan', lw=1, label='linear v')
    ax_t.plot(hw, '-', color='magenta', lw=1, label='angular w')
    ax_t.plot(he, '-', color='yellow', lw=1, label='ML error')
    ax_t.set_title("Telemetry (controls + ML learning curve)",
                   color='white', fontsize=10)
    ax_t.set_xlabel("step", color='white', fontsize=8)
    ax_t.tick_params(colors='white', labelsize=7)
    ax_t.legend(fontsize=8, loc='upper right', facecolor='#0b0f14',
                edgecolor='#445566', labelcolor='white')

    plt.tight_layout()
    plt.savefig("robot_report.png", dpi=110, facecolor='#0b0f14')
    print("\n[saved] robot_report.png")

    # --- Plot 2: Telemetry ---
    ax_t.plot(hv, '-', color='cyan', lw=1, label='linear v')
    ax_t.plot(hw, '-', color='magenta', lw=1, label='angular w')
    ax_t.plot(he, '-', color='yellow', lw=1, label='ML error')
    ax_t.set_title("Telemetry (controls + ML learning curve)",
                   color='white', fontsize=10)
    ax_t.set_xlabel("step", color='white', fontsize=8)
    ax_t.tick_params(colors='white', labelsize=7)
    ax_t.legend(fontsize=8, loc='upper right', facecolor='#0b0f14',
                edgecolor='#445566', labelcolor='white')

    plt.tight_layout()
    plt.savefig("robot_report.png", dpi=110, facecolor=fig.get_facecolor())
    print("\n[saved] robot_report.png")


if __name__ == "__main__":
    main()

"""
robot_core.py - Robotics algorithms library.
Subsystems: perception, CV, ML, optimization, manipulation,
motion planning, controls, autonomous navigation.
"""
import heapq
import numpy as np


def _inflate(grid, iters=1):
    g = grid.copy()
    for _ in range(iters):
        p = np.pad(g, 1, constant_values=False)
        g = (p[:-2,1:-1] | p[2:,1:-1] | p[1:-1,:-2] | p[1:-1,2:] |
             p[:-2,:-2] | p[:-2,2:] | p[2:,:-2] | p[2:,2:] | g)
    return g


class World:
    def __init__(self, w=40, h=30, res=1.0, inflate=1):
        self.width, self.height, self.res = w, h, res
        self.cols, self.rows = int(w/res), int(h/res)
        self.grid = np.zeros((self.rows, self.cols), dtype=bool)
        for (x, y, ww, hh) in [(10,5,4,12), (22,14,3,12), (28,4,6,5),
                               (6,20,8,3), (18,0,3,8)]:
            self.grid[int(y/res):int((y+hh)/res),
                      int(x/res):int((x+ww)/res)] = True
        self.grid = _inflate(self.grid, inflate)

    def is_free(self, c, r):
        return 0 <= c < self.cols and 0 <= r < self.rows and not self.grid[r, c]

    def w2g(self, x, y): return int(x/self.res), int(y/self.res)
    def g2w(self, c, r): return (c+0.5)*self.res, (r+0.5)*self.res


# ===================== PLANNING =====================
def astar(world, start, goal):
    sc, sr = world.w2g(*start); gc, gr = world.w2g(*goal)
    heap = [(0.0, (sc, sr))]; came = {}; g = {(sc, sr): 0.0}
    while heap:
        _, cur = heapq.heappop(heap)
        if cur == (gc, gr):
            path = [cur]
            while cur in came:
                cur = came[cur]; path.append(cur)
            return [world.g2w(c, r) for c, r in reversed(path)]
        c, r = cur
        for dc, dr in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nc, nr = c+dc, r+dr
            if not world.is_free(nc, nr): continue
            ng = g[cur] + (1.41421356 if dc and dr else 1.0)
            if (nc, nr) not in g or ng < g[(nc, nr)]:
                g[(nc, nr)] = ng; came[(nc, nr)] = cur
                heapq.heappush(heap, (ng + np.hypot(nc-gc, nr-gr), (nc, nr)))
    return None


def _seg_free(world, a, b, n=8):
    for t in np.linspace(0, 1, n):
        c, r = world.w2g(a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t)
        if not world.is_free(c, r): return False
    return True


def rrt(world, start, goal, max_iter=1500, step=1.0, goal_tol=2.0):
    nodes = [tuple(start)]; parents = {0: None}
    for _ in range(max_iter):
        q = goal if np.random.rand() < 0.15 else (
            np.random.uniform(0, world.width), np.random.uniform(0, world.height))
        arr = np.asarray(nodes)
        idx = int(np.argmin((arr[:,0]-q[0])**2 + (arr[:,1]-q[1])**2))
        near = nodes[idx]
        d = float(np.hypot(q[0]-near[0], q[1]-near[1]))
        if d < 1e-6: continue
        new = (near[0] + (q[0]-near[0])/d*min(step, d),
               near[1] + (q[1]-near[1])/d*min(step, d))
        if not _seg_free(world, near, new): continue
        nodes.append(new); parents[len(nodes)-1] = idx
        if np.hypot(new[0]-goal[0], new[1]-goal[1]) < goal_tol:
            path = [new]; k = len(nodes)-1
            while parents[k] is not None:
                k = parents[k]; path.append(nodes[k])
            return path[::-1]
    return None


# ===================== PERCEPTION =====================
def lidar_scan(world, x, y, n_rays=32, max_range=7.0, n_steps=16):
    a = np.linspace(0, 2*np.pi, n_rays, endpoint=False)
    ts = np.linspace(0, max_range, n_steps)
    hx = x + np.outer(np.cos(a), ts); hy = y + np.outer(np.sin(a), ts)
    c = np.clip((hx/world.res).astype(int), 0, world.cols-1)
    r = np.clip((hy/world.res).astype(int), 0, world.rows-1)
    occ = world.grid[r, c]
    hit = np.where(occ.any(axis=1), np.argmax(occ, axis=1), n_steps-1)
    rng = ts[hit]
    return rng, np.column_stack([x + np.cos(a)*rng, y + np.sin(a)*rng])


# ===================== COMPUTER VISION =====================
def camera_image(world, x, y, theta, size=16, res=1.0):
    n = int(size/res)
    i, j = np.mgrid[0:n, 0:n]
    fwd = i*res; left = (n/2.0 - j)*res
    fx, fy = np.cos(theta), np.sin(theta); lx, ly = -np.sin(theta), np.cos(theta)
    wx = x + fx*fwd + lx*left; wy = y + fy*fwd + ly*left
    c = np.clip((wx/world.res).astype(int), 0, world.cols-1)
    r = np.clip((wy/world.res).astype(int), 0, world.rows-1)
    inb = (wx >= 0) & (wx < world.width) & (wy >= 0) & (wy < world.height)
    img = np.zeros((n, n)); img[inb] = world.grid[r[inb], c[inb]]
    return img


def sobel_edges(img):
    p = np.pad(img, 1, mode='edge')
    gx = (p[:-2,2:] + 2*p[1:-1,2:] + p[2:,2:]) - \
         (p[:-2,:-2] + 2*p[1:-1,:-2] + p[2:,:-2])
    gy = (p[2:,:-2] + 2*p[2:,1:-1] + p[2:,2:]) - \
         (p[:-2,:-2] + 2*p[:-2,1:-1] + p[:-2,2:])
    m = np.hypot(gx, gy)
    return m/m.max() if m.max() > 0 else m


# ===================== ROBOT =====================
class Robot:
    def __init__(self, x, y, theta=0.0):
        self.x, self.y, self.theta = x, y, theta
        self.v = self.w = 0.0
        self.max_v, self.max_w = 3.0, 2.5
        self.L1, self.L2 = 1.2, 1.0
        self.q1 = self.q2 = 0.0
        self.gripper = 0.0
        self.has_object = False

    def step(self, v, w, dt, world):
        nth = (self.theta + w*dt + np.pi) % (2*np.pi) - np.pi
        nx = self.x + v*np.cos(self.theta)*dt
        ny = self.y + v*np.sin(self.theta)*dt
        c, r = world.w2g(nx, ny)
        if world.is_free(c, r):
            self.x, self.y, self.theta = nx, ny, nth
            self.v, self.w = v, w
        else:
            self.theta = nth; self.v, self.w = 0.0, w

    def arm_ik(self, tgt):
        dx, dy = tgt[0]-self.x, tgt[1]-self.y
        ct, st = np.cos(-self.theta), np.sin(-self.theta)
        tx, ty = ct*dx - st*dy, st*dx + ct*dy
        d = float(np.clip(np.hypot(tx, ty), 1e-3, self.L1+self.L2-1e-2))
        c2 = (d*d - self.L1**2 - self.L2**2) / (2*self.L1*self.L2)
        q2 = float(np.arccos(np.clip(c2, -1, 1)))
        q1 = float(np.arctan2(ty, tx) -
                   np.arctan2(self.L2*np.sin(q2), self.L1+self.L2*np.cos(q2)))
        return q1, q2

    def arm_points(self):
        a1 = self.theta + self.q1
        p1 = np.array([self.x + self.L1*np.cos(a1), self.y + self.L1*np.sin(a1)])
        a2 = a1 + self.q2
        p2 = p1 + self.L2*np.array([np.cos(a2), np.sin(a2)])
        return np.array([[self.x, self.y], p1, p2])


# ===================== CONTROLS =====================
def follow_path(robot, path, world, max_advance=4.0):
    """
    Sequential waypoint chaser with visibility check.
    Never jumps past an obstacle. Returns (v, w, target_idx).
    """
    arr = np.asarray(path)
    d2 = (arr[:,0]-robot.x)**2 + (arr[:,1]-robot.y)**2
    idx = int(np.argmin(d2))

    # advance while the NEXT waypoint is visible from current position
    while idx + 1 < len(path):
        nxt = path[idx + 1]
        dist_nxt = float(np.hypot(nxt[0]-robot.x, nxt[1]-robot.y))
        if dist_nxt > max_advance: break
        if not _seg_free(world, (robot.x, robot.y), nxt, n=4): break
        idx += 1

    target = path[idx]
    dx, dy = target[0]-robot.x, target[1]-robot.y
    dist = float(np.hypot(dx, dy))

    if idx == len(path)-1 and dist < 0.4:
        return 0.0, 0.0, idx

    heading_err = (np.arctan2(dy, dx) - robot.theta + np.pi) % (2*np.pi) - np.pi
    v = robot.max_v * max(0.2, 1.0 - abs(heading_err) / (np.pi/2))
    v = min(v, robot.max_v * 0.8)
    w = float(np.clip(2.0 * heading_err, -robot.max_w, robot.max_w))
    return v, w, idx


# ===================== OPTIMIZATION =====================
def mpc(world, robot, goal, K=30, H=6, dt=0.1):
    """Random-shooting MPC with structured + random candidates."""
    def cost(vs, ws):
        x, y, th = robot.x, robot.y, robot.theta
        c = 0.0
        for t in range(H):
            x += vs[t]*np.cos(th)*dt; y += vs[t]*np.sin(th)*dt; th += ws[t]*dt
            gc, gr = world.w2g(x, y)
            if not world.is_free(gc, gr): return np.inf
            c += 0.05*np.hypot(x-goal[0], y-goal[1]) + 0.05*ws[t]**2
        return c

    best_cost = np.inf
    best_u = (0.0, 0.0)

    # structured: constant turn rates at 3 forward speeds
    for w_try in np.linspace(-robot.max_w, robot.max_w, 9):
        for v_try in (0.3, 0.8, 1.5):
            vs = np.full(H, v_try); ws = np.full(H, w_try)
            c = cost(vs, ws)
            if c < best_cost: best_cost, best_u = c, (v_try, w_try)

    # random shots
    for _ in range(K):
        vs = np.random.uniform(0.2, robot.max_v, H)
        ws = np.random.uniform(-robot.max_w, robot.max_w, H)
        c = cost(vs, ws)
        if c < best_cost: best_cost, best_u = c, (float(vs[0]), float(ws[0]))

    return best_u if np.isfinite(best_cost) else (0.0, robot.max_w)


# ===================== ML =====================
class OnlineLinear:
    """Online LMS regressor for behavior cloning."""
    def __init__(self, n_in, n_out, lr=0.05):
        self.W = np.zeros((n_out, n_in)); self.b = np.zeros(n_out)
        self.lr = lr; self.last_err = 0.0

    def predict(self, x): return self.W @ x + self.b

    def update(self, x, y):
        err = y - self.predict(x)
        self.W += self.lr*np.outer(err, x); self.b += self.lr*err
        self.last_err = float(np.linalg.norm(err))
"""
synth_tum.py -- generate a synthetic TUM-format video sequence, only to run the whole pipeline end to end without real data

Scene: a 6x3x6 m room + a few boxes (all with random textures); the camera circles the room centre 1.2 times (with a loop),
with a pause (still segment) and a fast segment (with motion blur). Output has the same layout as TUM RGB-D:
    out/rgb/<ts>.png   out/rgb.txt   out/groundtruth.txt (tx ty tz qx qy qz qw, camera to world)
Usage: python synth_tum.py out_dir [--frames 600] [--W 640 --H 480]
"""
import argparse
import os

import cv2
import numpy as np

FX, FY = 525.0, 525.0


def make_texture(rng, n=512):
    t = np.zeros((n, n), np.float32)
    for s in (4, 8, 16, 32, 64):
        g = rng.random((s + 1, s + 1)).astype(np.float32)
        t += cv2.resize(g, (n, n), interpolation=cv2.INTER_CUBIC) / s ** 0.3
    t = (t - t.min()) / (t.max() - t.min())
    img = np.dstack([t * rng.uniform(0.4, 1.0) for _ in range(3)]) * 255
    for _ in range(60):
        c = tuple(int(x) for x in rng.integers(0, 255, 3))
        x, y = rng.integers(0, n, 2)
        if rng.random() < 0.5:
            w, h = rng.integers(8, 60, 2)
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), c, -1)
        else:
            cv2.circle(img, (int(x), int(y)), int(rng.integers(5, 30)), c, -1)
    return img.astype(np.uint8)


class Scene:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.room = (np.array([-3.0, -1.5, -3.0]), np.array([3.0, 1.5, 3.0]))
        self.boxes = [(np.array([-0.6, 0.7, -0.6]), np.array([0.6, 1.5, 0.6])),
                      (np.array([1.6, 0.5, 1.4]), np.array([2.4, 1.5, 2.2])),
                      (np.array([-2.5, 0.2, 1.0]), np.array([-1.8, 1.5, 2.0])),
                      (np.array([0.8, 1.0, -2.6]), np.array([2.2, 1.5, -2.0]))]
        n_obj = 1 + len(self.boxes)
        self.tex = [[make_texture(rng) for _ in range(6)] for _ in range(n_obj)]

    @staticmethod
    def _slab(o, d, lo, hi):
        inv = 1.0 / np.where(np.abs(d) < 1e-9, 1e-9, d)
        t0 = (lo - o) * inv
        t1 = (hi - o) * inv
        tmin = np.minimum(t0, t1)
        tmax = np.maximum(t0, t1)
        tn = tmin.max(axis=-1)
        tf = tmax.min(axis=-1)
        return tn, tf, tmin.argmax(axis=-1), tmax.argmin(axis=-1)

    def render(self, R, C, W, H):
        """R: camera-to-world rotation (columns = camera x right, y down, z forward), C: camera centre"""
        xs, ys = np.meshgrid((np.arange(W) - W / 2 + 0.5) / FX, (np.arange(H) - H / 2 + 0.5) / FY)
        d = np.stack([xs, ys, np.ones_like(xs)], -1) @ R.T
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        o = np.broadcast_to(C, d.shape)
        lo, hi = self.room
        _, tbest, _, axis = self._slab(o, d, lo, hi)
        obj = np.zeros(tbest.shape, np.int32)
        for k, (blo, bhi) in enumerate(self.boxes):
            tn, tf, ax, _ = self._slab(o, d, blo, bhi)
            hit = (tn <= tf) & (tn > 1e-3) & (tn < tbest)
            tbest = np.where(hit, tn, tbest)
            axis = np.where(hit, ax, axis)
            obj = np.where(hit, k + 1, obj)
        P = o + d * tbest[..., None]
        out = np.zeros((H, W, 3), np.uint8)
        for k in range(1 + len(self.boxes)):
            for a in range(3):
                for side in range(2):
                    sel = (obj == k) & (axis == a) & ((d[..., a] > 0) == bool(side))
                    if not sel.any():
                        continue
                    uv = np.delete(P[sel], a, axis=1)          # the two in-plane coordinates (metres)
                    tx = self.tex[k][a * 2 + side]
                    n = tx.shape[0]
                    u = (uv[:, 0] * 120.0) % n               # 1 m = 120 texels
                    v = (uv[:, 1] * 120.0) % n
                    x0, y0 = np.floor(u).astype(int), np.floor(v).astype(int)
                    fx, fy = (u - x0)[:, None], (v - y0)[:, None]
                    x1, y1 = (x0 + 1) % n, (y0 + 1) % n
                    c = (tx[y0, x0] * (1 - fx) * (1 - fy) + tx[y0, x1] * fx * (1 - fy) +
                         tx[y1, x0] * (1 - fx) * fy + tx[y1, x1] * fx * fy)
                    shade = 0.75 + 0.25 * (a == 1) + 0.1 * side
                    out[sel] = np.clip(c * shade, 0, 255).astype(np.uint8)
        return out


def look_rot(fwd, up=np.array([0.0, -1.0, 0.0])):
    z = fwd / np.linalg.norm(fwd)
    x = np.cross(-up, z)          # y points down
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], 1)


def trajectory(n):
    """Returns a continuous per-frame "progress" s in [0, 1.2] (turns), including pauses and fast segments"""
    speed = np.ones(n)
    speed[int(0.15 * n):int(0.25 * n)] = 0.03       # almost still
    speed[int(0.45 * n):int(0.52 * n)] = 3.0        # fast (blurred)
    speed[int(0.70 * n):int(0.78 * n)] = 0.05
    s = np.concatenate([[0], np.cumsum(speed[:-1])])
    return s / s[-1] * 1.2, speed


def pose(s):
    ang = 2 * np.pi * s
    C = np.array([1.3 * np.cos(ang), 0.1 * np.sin(3 * ang), 1.3 * np.sin(ang)])
    look = ang + 1.2 + 0.3 * np.sin(2 * ang)            # roughly facing the outer wall along the tangent
    fwd = np.array([np.cos(look), 0.05, np.sin(look)])
    return look_rot(fwd), C


def rot_to_quat(R):
    w = np.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    return np.array([(R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w), w])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--W", type=int, default=640)
    ap.add_argument("--H", type=int, default=480)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "rgb"), exist_ok=True)
    sc = Scene()
    s, speed = trajectory(a.frames)
    ds = np.gradient(s)
    rgb_lines, gt_lines = [], []
    for i in range(a.frames):
        ts = 1000.0 + i / 30.0
        sub = 5 if speed[i] > 2 else 1                  # fast segment: average of 5 sub-frames = motion blur
        acc = np.zeros((a.H, a.W, 3), np.float32)
        for k in range(sub):
            R, C = pose(s[i] + ds[i] * (k / sub - 0.5) * (sub > 1))
            acc += sc.render(R, C, a.W, a.H)
        img = (acc / sub).astype(np.uint8)
        name = f"{ts:.6f}.png"
        cv2.imwrite(os.path.join(a.out, "rgb", name), img[..., ::-1])
        R, C = pose(s[i])
        q = rot_to_quat(R)
        rgb_lines.append(f"{ts:.6f} rgb/{name}")
        gt_lines.append(f"{ts:.6f} " + " ".join(f"{x:.6f}" for x in (*C, *q)))
        if i % 50 == 0:
            print(f"frame {i}/{a.frames}", flush=True)
    with open(os.path.join(a.out, "rgb.txt"), "w") as f:
        f.write("# synthetic\n# timestamp filename\n" + "\n".join(rgb_lines) + "\n")
    with open(os.path.join(a.out, "groundtruth.txt"), "w") as f:
        f.write("# synthetic\n# timestamp tx ty tz qx qy qz qw\n" + "\n".join(gt_lines) + "\n")
    with open(os.path.join(a.out, "intrinsics.txt"), "w") as f:
        f.write(f"{FX} {FY} {a.W / 2} {a.H / 2}\n")


if __name__ == "__main__":
    main()

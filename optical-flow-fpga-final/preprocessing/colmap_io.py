"""colmap_io.py -- minimal reader for COLMAP binary models (cameras.bin / images.bin / points3D.bin), plus ATE evaluation"""
import os
import struct

import numpy as np

CAM_NPARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}


def _rd(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def read_cameras_bin(path):
    cams = {}
    with open(path, "rb") as f:
        n, = _rd(f, "Q")
        for _ in range(n):
            cid, model, w, h = _rd(f, "iiQQ")
            params = _rd(f, "d" * CAM_NPARAMS[model])
            cams[cid] = {"model": model, "width": w, "height": h, "params": params}
    return cams


def read_images_bin(path):
    imgs = {}
    with open(path, "rb") as f:
        n, = _rd(f, "Q")
        for _ in range(n):
            iid, qw, qx, qy, qz, tx, ty, tz, cid = _rd(f, "idddddddi")
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            n2d, = _rd(f, "Q")
            f.seek(n2d * 24, os.SEEK_CUR)
            imgs[iid] = {"name": name.decode(), "q": np.array([qw, qx, qy, qz]), "t": np.array([tx, ty, tz]),
                         "camera_id": cid, "n2d": n2d}
    return imgs


def count_points3d_bin(path):
    with open(path, "rb") as f:
        n, = _rd(f, "Q")
    return n


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def camera_centers(imgs):
    """name -> camera centre (world coordinates)"""
    return {im["name"]: -qvec2rotmat(im["q"]).T @ im["t"] for im in imgs.values()}


def model_stats(model_dir):
    imgs = read_images_bin(os.path.join(model_dir, "images.bin"))
    npts = count_points3d_bin(os.path.join(model_dir, "points3D.bin"))
    return {"images": len(imgs), "points": npts}


def best_model(sparse_dir):
    """the mapper may output several sub-models (0,1,2...); pick the one with the most registered images"""
    best, bestn = None, -1
    if not os.path.isdir(sparse_dir):
        return None, 0, 0
    subs = sorted(d for d in os.listdir(sparse_dir) if os.path.isfile(os.path.join(sparse_dir, d, "images.bin")))
    for d in subs:
        n = len(read_images_bin(os.path.join(sparse_dir, d, "images.bin")))
        if n > bestn:
            best, bestn = os.path.join(sparse_dir, d), n
    return best, bestn, len(subs)


def umeyama(src, dst, with_scale=True):
    """find s,R,t with dst ~= s R src + t (src/dst: Nx3)"""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = (D * np.diag(S)).sum() / var_s if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def read_tum_traj(path):
    ts, P = [], []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        v = line.split()
        ts.append(float(v[0]))
        P.append([float(x) for x in v[1:4]])
    return np.array(ts), np.array(P)


def ate(centers_by_ts, gt_ts, gt_pos, max_dt=0.02):
    """centers_by_ts: [(timestamp, centre)]; returns RMSE (m), number of matches, aligned errors"""
    est, ref = [], []
    for ts, c in centers_by_ts:
        j = np.searchsorted(gt_ts, ts)
        cand = [k for k in (j - 1, j) if 0 <= k < len(gt_ts)]
        if not cand:
            continue
        k = min(cand, key=lambda k: abs(gt_ts[k] - ts))
        if abs(gt_ts[k] - ts) <= max_dt:
            est.append(c)
            ref.append(gt_pos[k])
    if len(est) < 3:
        return {"rmse": None, "n": len(est)}
    est, ref = np.array(est), np.array(ref)
    s, R, t = umeyama(est, ref)
    err = np.linalg.norm((s * (R @ est.T)).T + t - ref, axis=1)
    return {"rmse": float(np.sqrt((err ** 2).mean())), "median": float(np.median(err)), "max": float(err.max()),
            "n": len(est), "scale": float(s), "traj_len": float(np.linalg.norm(np.diff(ref, axis=0), axis=1).sum())}

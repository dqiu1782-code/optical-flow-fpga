"""
hetero_prep.py -- heterogeneous 3DGS preprocessing pipeline: FPGA flow statistics -> CPU keyframe selection / pair pruning -> GPU features + matching -> CPU mapping

  stage           runs on         does
  decode          CPU             read all video frames (grayscale for the FPGA, originals for COLMAP)
  flow            FPGA            pyramid LK between neighbouring frames + on-chip per-frame statistics (mean motion, valid pixels, histogram) + coarsest-level thumbnail
                                  (results come from the Python model that is bit-exact with the RTL; time is the RTL cycle formula at 250 MHz)
  select          CPU (ARM)       pick keyframes by accumulated motion (fewer in still segments, more in fast ones), the sharpest frame in each window
  pairs           CPU (ARM)       candidate image pairs = neighbours within a motion window + thumbnail-ZNCC loop-closure candidates
  extract/match   GPU             COLMAP SIFT extraction + matching
  map             CPU             COLMAP incremental mapping

Methods compared (same number of keyframes K):
  B1  uniform frames + exhaustive matching            B2  uniform frames + sequential matching (overlap=10, no loop detection)
  O1  flow keyframes + exhaustive matching            O2  flow keyframes + flow-pruned candidate pairs (this work)

Usage (on Windows):
  python hetero_prep.py flow   --seq E:\\hetero\\data\\rgbd_dataset_freiburg3_long_office_household --work E:\\hetero\\work\\fr3
  python hetero_prep.py run    --work E:\\hetero\\work\\fr3 --methods B1 B2 O1 O2 --K 200 --colmap E:\\hetero\\colmap_env\\Library\\bin\\colmap.exe
  python hetero_prep.py report --work E:\\hetero\\work\\fr3
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "lk"), os.path.join(os.path.dirname(HERE), "python"), os.path.join(os.path.dirname(HERE), "optical_flow_fpga", "python")):
    if os.path.isfile(os.path.join(p, "lk_pyramid.py")):
        sys.path.insert(0, p)
        break
import lk_pyramid as P  # noqa: E402
import colmap_io as CIO  # noqa: E402

FCLK = 250e6
TUM_INTR = {"freiburg1": (517.3, 516.5, 318.6, 255.3), "freiburg2": (520.9, 521.0, 325.1, 249.7),
            "freiburg3": (535.4, 539.2, 320.1, 247.6)}


# ============================================================ read the sequence
def read_sequence(seq):
    """A TUM-format directory (rgb.txt) or a video file. Returns (timestamps, image paths, intrinsics or None)"""
    if os.path.isfile(seq):
        raise SystemExit("for a video file, extract frames first with --extract (see extract_video)")
    ts, paths = [], []
    for line in open(os.path.join(seq, "rgb.txt")):
        if line.startswith("#") or not line.strip():
            continue
        t, f = line.split()[:2]
        ts.append(float(t))
        paths.append(os.path.join(seq, f))
    intr = None
    if os.path.isfile(os.path.join(seq, "intrinsics.txt")):
        intr = tuple(float(x) for x in open(os.path.join(seq, "intrinsics.txt")).read().split()[:4])
    else:
        for k, v in TUM_INTR.items():
            if k in os.path.basename(os.path.normpath(seq)):
                intr = v
    return ts, paths, intr


def load_gray(path):
    import cv2
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


# ============================================================ FPGA stage (model)
def fpga_cycles_per_pair(W, H, levels=P.LEVELS):
    """Per-level RTL cycle formula (matches the tb_lk_pyr simulation): N_l + min(H_l,32)*W_l + 3*W_l + 47"""
    return sum(w * h + min(h, 32) * w + 3 * w + 47 for w, h in P.level_dims(W, H, levels))


def fpga_build_cycles(W, H, levels=P.LEVELS):
    """Pyramid build: once per frame, input stream 1 pixel/cycle, all downsampling levels in parallel -> about W*H cycles (done as the frame arrives)"""
    return W * H


def _flow_job(args):
    i, p0, p1 = args
    I0, I1 = load_gray(p0), load_gray(p1)
    u, v, f = P.lk_pyramid_fixed(I0, I1, du_max=P.DU_MAX)
    st = P.frame_stats(u, v, f)
    st["sharp"] = P.frame_sharp(I0)              # lk_sharp: sharpness computed while the pyramid is built
    thumb = P.build_pyramid(I0, P.LEVELS)[-1]
    return i, st, thumb


def _sharp_job(p):
    import cv2
    g = load_gray(p).astype(np.float32)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def cmd_flow(a):
    os.makedirs(a.work, exist_ok=True)
    ts, paths, intr = read_sequence(a.seq)
    if a.max_frames:
        ts, paths = ts[:a.max_frames], paths[:a.max_frames]
    N = len(paths)
    t0 = time.perf_counter()
    I = load_gray(paths[0])
    H, W = I.shape
    with Pool(a.jobs) as pool:
        # decode: read every frame once (a common cost for video input, counted for every method)
        td = time.perf_counter()
        lap = pool.map(_sharp_job, paths, chunksize=16)      # also compute the CPU reference sharpness (only used to choose test frames)
        t_decode = time.perf_counter() - td
        jobs = [(i, paths[i], paths[i + 1]) for i in range(N - 1)]
        res = [None] * (N - 1)
        thumbs = [None] * N
        tf = time.perf_counter()
        for k, (i, st, th) in enumerate(pool.imap_unordered(_flow_job, jobs, chunksize=8)):
            res[i] = st
            thumbs[i] = th
            if k % 200 == 0:
                print(f"  flow {k}/{N - 1}", flush=True)
        thumbs[N - 1] = P.build_pyramid(load_gray(paths[N - 1]), P.LEVELS)[-1]
        t_emul = time.perf_counter() - tf
    cyc = fpga_cycles_per_pair(W, H)
    meta = {"seq": os.path.abspath(a.seq), "N": N, "W": W, "H": H, "intrinsics": intr,
            "cycles_per_pair": cyc, "build_cycles_per_frame": fpga_build_cycles(W, H),
            "fpga_time_1lane_s": (N - 1) * cyc / FCLK, "fpga_lanes_k26": 10,
            "fpga_time_10lane_s": (N - 1) * cyc / FCLK / 10,
            "fpga_bytes_in": N * W * H, "fpga_bytes_out": (N - 1) * 64 + N * (thumbs[0].size + 5),
            "decode_s": t_decode, "python_model_s": t_emul, "jobs": a.jobs, "total_s": time.perf_counter() - t0}
    np.savez_compressed(os.path.join(a.work, "flow.npz"), ts=np.array(ts), paths=np.array(paths),
                        count=np.array([r["count"] for r in res]), sum_u=np.array([r["sum_u"] for r in res]),
                        sum_v=np.array([r["sum_v"] for r in res]), sum_l1=np.array([r["sum_l1"] for r in res]),
                        max_l1=np.array([r["max_l1"] for r in res]), hist=np.array([r["hist"] for r in res]),
                        sharp=np.array([r["sharp"] for r in res] + [P.frame_sharp(load_gray(paths[N - 1]))]),
                        thumbs=np.stack(thumbs), lap=np.array(lap))
    json.dump(meta, open(os.path.join(a.work, "flow_meta.json"), "w"), indent=2)
    print(json.dumps({k: meta[k] for k in ("N", "W", "H", "cycles_per_pair", "fpga_time_1lane_s",
                                           "fpga_time_10lane_s", "decode_s", "python_model_s")}, indent=1))


# ============================================================ CPU (ARM) stage: keyframe selection / pruning
def load_flow(work):
    d = dict(np.load(os.path.join(work, "flow.npz")))
    meta = json.load(open(os.path.join(work, "flow_meta.json")))
    cnt = np.maximum(d["count"], 1)
    d["motion"] = d["sum_l1"] / cnt / 256.0                     # mean |u|+|v| per pair (pixels)
    d["meanvec"] = np.hypot(d["sum_u"], d["sum_v"]) / cnt / 256.0
    d["valid_frac"] = d["count"] / float(meta["W"] * meta["H"])
    return d, meta


def pick_test_frames(N, lap, block=50, lo=15, hi=35):
    """Evaluation test frames: one per 50-frame block, the frame with the largest CPU Laplacian variance near the block centre (shared by all methods; no training frame within +-3 frames)"""
    test = []
    for b in range(N // block):
        s, e = b * block + lo, min(N, b * block + hi)
        if s < e:
            test.append(s + int(np.argmax(lap[s:e])))
    banned = np.zeros(N, bool)
    for t in test:
        banned[max(0, t - 3):t + 4] = True
    return test, banned


def uniform_keyframes(N, K, banned):
    allowed = np.flatnonzero(~banned)
    out = []
    for t in np.linspace(0, N - 1, K):
        order = allowed[np.argsort(np.abs(allowed - t))]
        for j in order:
            if j not in out:
                out.append(int(j))
                break
    return sorted(out)


def flow_keyframes(d, K, banned, win_frac=0.35):
    """Split the accumulated motion into K equal segments and, near each boundary (+-win_frac of a segment), pick the frame with the largest FPGA sharpness (lk_sharp)"""
    m = d["motion"]
    N = len(m) + 1
    S = np.concatenate([[0.0], np.cumsum(m)])
    seg = S[-1] / max(K - 1, 1)
    # per-frame sharpness score: lk_sharp's sum(dx²+dy²) (motion blur -> weaker fine-scale gradients -> lower score)
    score = d["sharp"].astype(float)
    out = []
    for t in np.linspace(0, S[-1], K):
        cand = np.flatnonzero((np.abs(S - t) <= win_frac * seg) & ~banned)
        if cand.size == 0:
            c2 = np.flatnonzero(~banned)
            cand = c2[np.argsort(np.abs(S[c2] - t))[:3]]
        cand = [c for c in cand if c not in out]
        if not cand:
            continue
        out.append(int(max(cand, key=lambda c: score[c])))
    return sorted(set(out)), S


def zncc_shift(A, B, r=2):
    """Thumbnail ZNCC, allowing +-r pixel shifts; returns the maximum. A,B: (h,w) uint8"""
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    h, w = A.shape
    best = -1.0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            a = A[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)]
            b = B[max(0, -dy):h + min(0, -dy), max(0, -dx):w + min(0, -dx)]
            a = a - a.mean()
            b = b - b.mean()
            den = math.sqrt(float((a * a).sum() * (b * b).sum())) + 1e-6
            best = max(best, float((a * b).sum()) / den)
    return best


def zncc_matrix(th, r=2):
    """Pairwise ZNCC of all thumbnails (max over +-r pixel shifts), one matrix multiply per shift. th: (K,h,w) -> (K,K)"""
    th = th.astype(np.float32)
    K, h, w = th.shape
    best = np.full((K, K), -1.0, np.float32)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            a = th[:, max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)].reshape(K, -1)
            b = th[:, max(0, -dy):h + min(0, -dy), max(0, -dx):w + min(0, -dx)].reshape(K, -1)
            a = a - a.mean(1, keepdims=True)
            b = b - b.mean(1, keepdims=True)
            a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-6
            b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-6
            best = np.maximum(best, a @ b.T)
    return np.maximum(best, best.T)


def prune_pairs(kf, S, thumbs, W, win=0.5, nmin=4, loop_k=3, loop_thr=0.6, r=2):
    """Candidate pairs = neighbours with accumulated motion < win*W (at least nmin sequential neighbours) + the loop_k distant frames with the highest thumbnail ZNCC"""
    K = len(kf)
    Sk = S[kf]
    pairs, loops = set(), []
    for i in range(K):
        for j in range(i + 1, K):
            if j - i <= nmin or Sk[j] - Sk[i] <= win * W:
                pairs.add((i, j))
            else:
                break
    Z = zncc_matrix(thumbs[kf], r)
    for i in range(K):
        far = np.flatnonzero(np.abs(Sk - Sk[i]) > win * W)
        if far.size == 0:
            continue
        for j in far[np.argsort(-Z[i, far])][:loop_k]:
            if Z[i, j] >= loop_thr:
                p = (min(i, int(j)), max(i, int(j)))
                if p not in pairs:
                    pairs.add(p)
                    loops.append((p[0], p[1], round(float(Z[i, j]), 3)))
    return sorted(pairs), loops


# ============================================================ COLMAP (GPU extraction/matching, CPU mapping)
class Colmap:
    def __init__(self, exe, gpu=True):
        self.exe = exe
        self.gpu = gpu
        self.env = dict(os.environ)
        bindir = os.path.dirname(os.path.abspath(exe)) if os.path.sep in exe or "/" in exe else ""
        if bindir:
            extra = [bindir]
            root = os.path.dirname(os.path.dirname(bindir))   # conda env root (Windows: env\Library\bin)
            extra += [os.path.join(root, "Library", "bin"), os.path.join(root, "Library", "mingw-w64", "bin"),
                      os.path.join(root, "Library", "usr", "bin"), os.path.join(root, "Scripts"), root]
            self.env["PATH"] = os.pathsep.join(extra + [self.env.get("PATH", "")])
        self.help = {}

    def opts(self, cmd):
        if cmd not in self.help:
            r = subprocess.run([self.exe, cmd, "-h"], capture_output=True, text=True, env=self.env)
            self.help[cmd] = r.stdout + r.stderr
        return self.help[cmd]

    def has(self, cmd, opt):
        return ("--" + opt) in self.opts(cmd)

    def pick(self, cmd, *cands):
        for c in cands:
            if self.has(cmd, c):
                return c
        return None

    def run(self, cmd, args, log):
        full = [self.exe, cmd] + [str(x) for x in args]
        t0 = time.perf_counter()
        with open(log, "a", encoding="utf-8") as f:
            f.write("\n$ " + " ".join(full) + "\n")
            f.flush()
            r = subprocess.run(full, stdout=f, stderr=subprocess.STDOUT, env=self.env)
        dt = time.perf_counter() - t0
        if r.returncode != 0:
            raise RuntimeError(f"colmap {cmd} failed (exit {r.returncode}), see {log}")
        return dt

    def gpu_args(self, cmd, kind):
        o = self.pick(cmd, f"Feature{kind}.use_gpu", f"Sift{kind}.use_gpu")
        return [f"--{o}", 1 if self.gpu else 0] if o else []


def link_or_copy(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def method_frames(d, meta, method, K, banned):
    N = meta["N"]
    if method in ("B1", "B2"):
        return uniform_keyframes(N, K, banned), None
    kf, S = flow_keyframes(d, K, banned)
    return kf, S


def run_method(a, method, d, meta, test, banned, cm):
    """The method may carry a variant suffix: O2@1.0 = O2 with pruning window win=1.0 (results go to <work>/O2@1.0/)"""
    t_all = time.perf_counter()
    mdir = os.path.join(a.work, method)
    win = a.win
    if "@" in method:
        method, win = method.split("@")[0], float(method.split("@")[1])
    if os.path.isdir(mdir) and a.fresh:
        shutil.rmtree(mdir)
    os.makedirs(mdir, exist_ok=True)
    log = os.path.join(mdir, "colmap_log.txt")
    timing = {}
    paths, ts = d["paths"], d["ts"]
    W = meta["W"]

    t0 = time.perf_counter()
    kf, S = method_frames(d, meta, method, a.K, banned)
    pairs, loops = None, []
    if method == "O2":
        pairs, loops = prune_pairs(kf, S, d["thumbs"], W, win=win, nmin=a.nmin, loop_k=a.loop_k, loop_thr=a.loop_thr)
    timing["select_cpu_s"] = time.perf_counter() - t0
    K = len(kf)
    names = [os.path.basename(str(paths[i])) for i in kf]
    tnames = [os.path.basename(str(paths[i])) for i in test]

    img_dir = os.path.join(mdir, "images")
    os.makedirs(img_dir, exist_ok=True)
    for i in list(kf) + list(test):
        link_or_copy(str(paths[i]), os.path.join(img_dir, os.path.basename(str(paths[i]))))
    with open(os.path.join(mdir, "train_list.txt"), "w") as f:
        f.write("\n".join(names) + "\n")
    with open(os.path.join(mdir, "test_list.txt"), "w") as f:
        f.write("\n".join(tnames) + "\n")

    db = os.path.join(mdir, "database.db")
    if os.path.exists(db):
        os.remove(db)
    intr = meta.get("intrinsics")
    cam = ["--ImageReader.single_camera", 1]
    if intr:
        cam += ["--ImageReader.camera_model", "PINHOLE", "--ImageReader.camera_params", ",".join(f"{x}" for x in intr)]
    else:
        cam += ["--ImageReader.camera_model", "SIMPLE_RADIAL"]
    # ---- GPU: feature extraction
    timing["extract_s"] = cm.run("feature_extractor", ["--database_path", db, "--image_path", img_dir,
                                                       "--image_list_path", os.path.join(mdir, "train_list.txt")]
                                 + cam + cm.gpu_args("feature_extractor", "Extraction"), log)
    # ---- GPU: matching
    if method in ("B1", "O1"):
        n_pairs = K * (K - 1) // 2
        timing["match_s"] = cm.run("exhaustive_matcher", ["--database_path", db] + cm.gpu_args("exhaustive_matcher", "Matching"), log)
    elif method == "B2":
        ov = a.seq_overlap
        n_pairs = sum(min(ov, K - 1 - i) for i in range(K))
        args = ["--database_path", db, "--SequentialMatching.overlap", ov]
        if cm.has("sequential_matcher", "SequentialMatching.loop_detection"):
            args += ["--SequentialMatching.loop_detection", 0]
        timing["match_s"] = cm.run("sequential_matcher", args + cm.gpu_args("sequential_matcher", "Matching"), log)
    else:
        pf = os.path.join(mdir, "pairs.txt")
        with open(pf, "w") as f:
            for i, j in pairs:
                f.write(f"{names[i]} {names[j]}\n")
        n_pairs = len(pairs)
        timing["match_s"] = cm.run("matches_importer", ["--database_path", db, "--match_list_path", pf, "--match_type", "pairs"]
                                   + cm.gpu_args("matches_importer", "Matching"), log)
    # ---- CPU: mapping (intrinsics are calibrated and kept fixed, the same for every method)
    sp = os.path.join(mdir, "sparse")
    if os.path.isdir(sp):
        shutil.rmtree(sp)
    os.makedirs(sp)
    margs = ["--database_path", db, "--image_path", img_dir, "--output_path", sp]
    mcmd, pre = ("global_mapper", "GlobalMapper") if a.mapper == "global" else ("mapper", "Mapper")
    if intr:
        margs += [f"--{pre}.ba_refine_focal_length", 0, f"--{pre}.ba_refine_principal_point", 0,
                  f"--{pre}.ba_refine_extra_params", 0]
    timing["map_s"] = cm.run(mcmd, margs, log)
    best, nreg, nmodels = CIO.best_model(sp)
    timing["prep_total_s"] = timing["select_cpu_s"] + timing["extract_s"] + timing["match_s"] + timing["map_s"]

    res = {"mapper": a.mapper, "method": os.path.basename(mdir), "win": win if method == "O2" else None, "K": K, "keyframes": [int(i) for i in kf], "n_pairs": int(n_pairs),
           "n_loops": len(loops), "loops": loops[:200], "timing": timing, "registered": nreg, "n_models": nmodels,
           "best_model": best}
    if best:
        res["points3d"] = CIO.count_points3d_bin(os.path.join(best, "points3D.bin"))
        # ---- pose accuracy: ATE against the ground-truth trajectory
        gt = os.path.join(meta["seq"], "groundtruth.txt")
        if os.path.isfile(gt):
            gts, gtp = CIO.read_tum_traj(gt)
            cen = CIO.camera_centers(CIO.read_images_bin(os.path.join(best, "images.bin")))
            name2ts = {os.path.basename(str(p)): float(t) for p, t in zip(paths, ts)}
            pts = sorted((name2ts[n], c) for n, c in cen.items() if n in name2ts)
            res["ate"] = CIO.ate(pts, gts, gtp)
        # ---- evaluation: register the test frames into the model (not counted in preprocessing time)
        try:
            res["test"] = register_test(cm, mdir, db, img_dir, best, names, tnames, kf, test, cam, log, intr)
        except RuntimeError as e:
            res["test"] = {"error": str(e)}
    res["wall_s"] = time.perf_counter() - t_all
    json.dump(res, open(os.path.join(mdir, "result.json"), "w"), indent=2)
    return res


def register_test(cm, mdir, db, img_dir, best, names, tnames, kf, test, cam, log, intr):
    t0 = time.perf_counter()
    ex = ["--database_path", db, "--image_path", img_dir, "--image_list_path", os.path.join(mdir, "test_list.txt")]
    if cm.has("feature_extractor", "ImageReader.existing_camera_id"):
        ex += ["--ImageReader.existing_camera_id", 1]
    else:
        ex += cam
    cm.run("feature_extractor", ex + cm.gpu_args("feature_extractor", "Extraction"), log)
    # each test frame is paired with its 10 temporally nearest training keyframes (same rule for all methods)
    kf = np.array(kf)
    pf = os.path.join(mdir, "test_pairs.txt")
    with open(pf, "w") as f:
        for t, tn in zip(test, tnames):
            for j in np.argsort(np.abs(kf - t))[:10]:
                f.write(f"{tn} {names[j]}\n")
    cm.run("matches_importer", ["--database_path", db, "--match_list_path", pf, "--match_type", "pairs"]
           + cm.gpu_args("matches_importer", "Matching"), log)
    reg = os.path.join(mdir, "sparse_reg")
    if os.path.isdir(reg):
        shutil.rmtree(reg)
    os.makedirs(reg)
    rargs = ["--database_path", db, "--input_path", best, "--output_path", reg]
    for o in ("Mapper.abs_pose_refine_focal_length", "Mapper.abs_pose_refine_extra_params", "Mapper.ba_refine_focal_length", "Mapper.ba_refine_extra_params", "Mapper.ba_refine_principal_point"):
        if cm.has("image_registrator", o):
            rargs += ["--" + o, 0]
    cm.run("image_registrator", rargs, log)
    imgs = CIO.read_images_bin(os.path.join(reg, "images.bin"))
    regnames = {im["name"] for im in imgs.values()}
    ok = [n for n in tnames if n in regnames]
    # ---- 3DGS scene directory: poses from the registered model, initial cloud from the training-only model (test frames must not leak into initialisation)
    scene = os.path.join(mdir, "scene")
    if os.path.isdir(scene):
        shutil.rmtree(scene)
    os.makedirs(os.path.join(scene, "sparse", "0"))
    os.makedirs(os.path.join(scene, "images"))
    shutil.copy2(os.path.join(reg, "cameras.bin"), os.path.join(scene, "sparse", "0", "cameras.bin"))
    shutil.copy2(os.path.join(reg, "images.bin"), os.path.join(scene, "sparse", "0", "images.bin"))
    shutil.copy2(os.path.join(best, "points3D.bin"), os.path.join(scene, "sparse", "0", "points3D.bin"))
    for n in regnames:
        link_or_copy(os.path.join(img_dir, n), os.path.join(scene, "images", n))
    with open(os.path.join(scene, "sparse", "0", "test.txt"), "w") as f:
        f.write("\n".join(ok) + "\n")
    return {"n_test": len(tnames), "registered": len(ok), "scene": scene, "time_s": time.perf_counter() - t0}


def cmd_run(a):
    d, meta = load_flow(a.work)
    test, banned = pick_test_frames(meta["N"], d["lap"])
    json.dump({"test": test}, open(os.path.join(a.work, "test_frames.json"), "w"))
    cm = Colmap(a.colmap, gpu=not a.cpu)
    allres = {}
    for m in a.methods:
        print(f"===== {m} =====", flush=True)
        try:
            r = run_method(a, m, d, meta, test, banned, cm)
        except Exception as e:                       # one method failing does not stop the others
            print(f"  {m} failed: {e}", flush=True)
            continue
        allres[m] = r
        t = r["timing"]
        print(f"  K={r['K']} pairs={r['n_pairs']} loops={r['n_loops']} registered={r['registered']}/{r['K']} "
              f"models={r['n_models']} | extract {t['extract_s']:.1f}s match {t['match_s']:.1f}s map {t['map_s']:.1f}s"
              f" | ATE {r.get('ate', {}).get('rmse')}", flush=True)
    return allres


def cmd_select(a):
    """Keyframe selection / pruning only (no COLMAP), for tuning and plots"""
    d, meta = load_flow(a.work)
    test, banned = pick_test_frames(meta["N"], d["lap"])
    out = {"test": test}
    for m in a.methods:
        kf, S = method_frames(d, meta, m, a.K, banned)
        r = {"kf": kf}
        if m == "O2":
            t0 = time.perf_counter()
            pairs, loops = prune_pairs(kf, S, d["thumbs"], meta["W"], win=a.win, nmin=a.nmin, loop_k=a.loop_k,
                                       loop_thr=a.loop_thr)
            r.update(pairs=len(pairs), loops=loops, prune_s=time.perf_counter() - t0)
        out[m] = r
        print(m, len(kf), {k: v for k, v in r.items() if k not in ("kf", "loops")}, "loops:", len(r.get("loops", [])))
    json.dump(out, open(os.path.join(a.work, "select_preview.json"), "w"), indent=1)


def cmd_report(a):
    meta = json.load(open(os.path.join(a.work, "flow_meta.json")))
    rows = []
    for m in sorted(os.listdir(a.work)):
        p = os.path.join(a.work, m, "result.json")
        if not os.path.isfile(p):
            continue
        r = json.load(open(p))
        t = r["timing"]
        fpga = meta["fpga_time_10lane_s"] if m.startswith("O") else 0.0
        g = {}
        gp = os.path.join(a.work, m, "gs_result.json")
        if os.path.isfile(gp):
            g = json.load(open(gp))
        ate = (r.get("ate") or {}).get("rmse")
        rows.append({"method": m, "K": r["K"], "registered": r["registered"], "models": r["n_models"],
                     "pairs": r["n_pairs"], "loops": r["n_loops"], "fpga_s": fpga, "select_s": t["select_cpu_s"],
                     "extract_s": t["extract_s"], "match_s": t["match_s"], "map_s": t["map_s"],
                     "prep_total_s": t["prep_total_s"] + fpga, "ate_m": ate,
                     "test_reg": (r.get("test") or {}).get("registered"), "points": r.get("points3d"),
                     "psnr": g.get("psnr"), "ssim": g.get("ssim"), "lpips": g.get("lpips")})
    keys = list(rows[0].keys()) if rows else []
    print("  ".join(keys))
    for row in rows:
        print("  ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row.values()))
    json.dump({"meta": meta, "rows": rows}, open(os.path.join(a.work, "report.json"), "w"), ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("flow")
    f.add_argument("--seq", required=True)
    f.add_argument("--work", required=True)
    f.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    f.add_argument("--max_frames", type=int, default=0)
    for name in ("run", "select"):
        r = sub.add_parser(name)
        r.add_argument("--work", required=True)
        r.add_argument("--methods", nargs="+", default=["B1", "B2", "O1", "O2"])
        r.add_argument("--K", type=int, default=200)
        r.add_argument("--win", type=float, default=0.5, help="pruning window: accumulated motion < win x image width")
        r.add_argument("--nmin", type=int, default=4)
        r.add_argument("--loop_k", type=int, default=3)
        r.add_argument("--loop_thr", type=float, default=0.6)
        r.add_argument("--seq_overlap", type=int, default=10)
        r.add_argument("--colmap", default="colmap")
        r.add_argument("--cpu", action="store_true", help="COLMAP without GPU")
        r.add_argument("--fresh", action="store_true")
        r.add_argument("--mapper", default="incremental", choices=["incremental", "global"],
                       help="incremental = COLMAP mapper; global = COLMAP 4 global_mapper (GLOMAP)")
    rp = sub.add_parser("report")
    rp.add_argument("--work", required=True)
    a = ap.parse_args()
    {"flow": cmd_flow, "run": cmd_run, "select": cmd_select, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()

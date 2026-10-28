"""
stream_prep.py -- streaming preprocessing with FPGA, GPU and CPU working at the same time (vs hetero_prep.py, where each stage finishes before the next starts)

  thread                hardware      does
  camera                sensor        hands out frames at 30 fps (online) or all at once (offline); dispatches neighbouring frame pairs to the flow devices
  fpga                  FPGA          pyramid LK + statistics. Results = flow.npz precomputed with the bit-exact model,
                                      time = RTL cycle formula (10 lanes @ 250 MHz, queued per lane, waiting for frames to arrive)
  gpu_flow              GPU           float32 version of the same algorithm (gpu_flow.py), really run on the 5090
  select                ARM (CPU)     causal keyframe selection (only statistics that have arrived, with a small look-back window of +-35% of a segment) + candidate pairs
  colmap_gpu            GPU           extracts SIFT incrementally for new keyframes (appended to the database) and matches new candidate pairs incrementally
  mapper                CPU           each time a batch of new keyframes is ready, continues incremental mapping from the previous model with --input_path

Flow work split (--flow):
  fpga            all frame pairs to the FPGA
  gpu             all to the GPU
  split-static    static split by throughput ratio (--gpu_share)
  split-dynamic   one shared task queue; whichever of FPGA and GPU is idle takes the next item (work stealing, automatic load balancing)

--serial: same threads, same keyframes/candidate pairs, but each stage waits for the previous one to finish completely -- to measure what running concurrently saves.
Outputs: <out>/stream_result.json (stage completion times, device busy times, ATE ...) and timeline.json (for Gantt charts).
"""
import argparse
import json
import math
import os
import queue
import shutil
import sqlite3
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hetero_prep as HP  # noqa: E402
import colmap_io as CIO  # noqa: E402

LANES = 10


class Timeline:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.ev = []
        self.lock = threading.Lock()
        self.marks = {}

    def now(self):
        return time.perf_counter() - self.t0

    def sleep_until(self, t):
        d = t - self.now()
        if d > 0:
            time.sleep(d)

    def add(self, lane, s, e, label=""):
        with self.lock:
            self.ev.append((lane, round(s, 4), round(e, 4), label))

    def mark(self, k):
        self.marks[k] = round(self.now(), 4)


class Stream:
    """State shared by all threads"""

    def __init__(self, a, d, meta, banned, test):
        self.a, self.d, self.meta = a, d, meta
        self.N = meta["N"] if not a.max_frames else min(meta["N"], a.max_frames)
        self.W, self.H = meta["W"], meta["H"]
        self.banned, self.test = banned, test
        self.tl = Timeline()
        self.arrive = [i / a.fps if a.rate == "online" else 0.0 for i in range(self.N)]
        self.cond = threading.Condition()
        self.stats = {}                # frame-pair index -> statistics
        self.src = {}                  # frame-pair index -> "fpga"/"gpu"
        self.flow_done = False
        # keyframes / candidate pairs
        self.kf = []
        self.kf_time = []
        self.pending_imgs = []
        self.pending_pairs = []
        self.all_pairs = []
        self.n_loops = 0
        self.select_done = False
        # COLMAP
        self.db_lock = threading.Lock()
        self.extracted = set()
        self.matched_imgs = set()
        self.gpu_done = False
        self.model = None
        self.errors = []
        self.frames = None
        self.paths = [str(p) for p in d["paths"][:self.N]]
        self.names = [os.path.basename(p) for p in self.paths]


# ============================================================ camera + flow
def camera(st, queues, assign):
    """Dispatch frame pair (i-1, i) when frame i arrives. assign(i) -> queue index; offline, 32 pairs at a time"""
    tl, a = st.tl, st.a
    chunk = {}
    csize = 1 if a.rate == "online" else 3 * LANES
    for i in range(1, st.N):
        tl.sleep_until(st.arrive[i])
        q = assign(i - 1)
        chunk.setdefault(q, []).append(i - 1)
        if len(chunk[q]) >= csize:
            queues[q].put(chunk.pop(q))
    for q, c in chunk.items():
        queues[q].put(c)
    for q in queues:
        q.put(None)
    tl.mark("capture_end")


def emit(st, i, rec, src):
    with st.cond:
        st.stats[i] = rec
        st.src[i] = src
        st.cond.notify_all()


def fpga_worker(st, q):
    d, tl = st.d, st.tl
    dur = HP.fpga_cycles_per_pair(st.W, st.H) / HP.FCLK
    free = [0.0] * LANES
    busy = [0.0] * LANES
    while True:
        chunk = q.get()
        if chunk is None:
            q.put(None)
            break
        sched = []
        now = tl.now()
        for i in chunk:
            l = int(np.argmin(free))
            s = max(free[l], st.arrive[i + 1], now)
            free[l] = s + dur
            busy[l] += dur
            sched.append((s + dur, s, l, i))
        for e, s, l, i in sorted(sched):
            tl.sleep_until(e)
            rec = {"count": int(d["count"][i]), "sum_l1": float(d["sum_l1"][i]), "sum_u": float(d["sum_u"][i]),
                   "sum_v": float(d["sum_v"][i]), "sharp": float(d["sharp"][i]), "thumb": d["thumbs"][i]}
            emit(st, i, rec, "fpga")
        # Gantt chart: one segment per chunk per lane
        for l in range(LANES):
            seg = [(s, e) for e, s, ll, _ in sched if ll == l]
            if seg:
                tl.add(f"FPGA lane{l}", min(x[0] for x in seg), max(x[1] for x in seg), f"{len(seg)} pairs")
    st.fpga_busy = busy


def gpu_flow_worker(st, q):
    import torch
    import gpu_flow as GF
    tl = st.tl
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    busy = 0.0
    while True:
        chunk = q.get()
        if chunk is None:
            q.put(None)
            break
        s = tl.now()
        idx = np.array(chunk)
        I0 = torch.from_numpy(st.frames[idx]).to(dev, non_blocking=True)
        I1 = torch.from_numpy(st.frames[idx + 1]).to(dev, non_blocking=True)
        recs = GF.pyr_flow_stats(I0, I1)
        e = tl.now()
        busy += e - s
        tl.add("GPU flow", s, e, f"{len(chunk)} pairs")
        for i, r in zip(chunk, recs):
            emit(st, i, r, "gpu")
    st.gpu_flow_busy = busy


# ============================================================ ARM: causal keyframe selection + candidate pairs
def zncc_one(t, T, r=2):
    """ZNCC of one thumbnail against several (K,h,w) (max over +-r shifts)"""
    t = t.astype(np.float32)
    T = T.astype(np.float32)
    h, w = t.shape
    best = np.full(len(T), -1.0, np.float32)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            a = t[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)].ravel()
            b = T[:, max(0, -dy):h + min(0, -dy), max(0, -dx):w + min(0, -dx)].reshape(len(T), -1)
            a = a - a.mean()
            b = b - b.mean(1, keepdims=True)
            z = (b @ a) / (np.linalg.norm(b, axis=1) * np.linalg.norm(a) + 1e-6)
            best = np.maximum(best, z)
    return best


def selector(st):
    a, tl = st.a, st.tl
    method = a.method
    W = st.W
    S = [0.0]
    sharp, thumbs = {}, {}
    T = st.T
    w = 0.35
    cand, last_S = [], None
    t_busy = 0.0
    step = (st.N - 1) / max(a.K - 1, 1)
    next_uniform = 0.0

    def add_kf(j):
        nonlocal t_busy
        s0 = time.perf_counter()
        k = len(st.kf)
        pairs = []
        if method in ("B1", "O1"):
            pairs = [(i, k) for i in range(k)]
        elif method == "B2":
            pairs = [(i, k) for i in range(max(0, k - a.seq_overlap), k)]
        else:  # O2
            Sk = [S[x] for x in st.kf]
            for i in range(k - 1, -1, -1):
                if k - i <= a.nmin or S[j] - Sk[i] <= a.win * W:
                    pairs.append((i, k))
                else:
                    break
            far = [i for i in range(k) if abs(S[j] - Sk[i]) > a.win * W]
            if far:
                z = zncc_one(thumbs[j], np.stack([thumbs[st.kf[i]] for i in far]))
                for o in np.argsort(-z)[:a.loop_k]:
                    if z[o] >= a.loop_thr and (far[o], k) not in pairs:
                        pairs.append((far[o], k))
                        st.n_loops += 1
        with st.cond:
            st.kf.append(j)
            st.kf_time.append(tl.now())
            st.pending_imgs.append(st.names[j])
            pp = [(st.names[st.kf[i]], st.names[st.kf[kk]]) for i, kk in pairs]
            st.pending_pairs.extend(pp)
            st.all_pairs.extend(pp)
            st.cond.notify_all()
        t_busy += time.perf_counter() - s0

    if a.serial and method not in ("B1", "B2"):      # serial baseline: wait for all flow before selecting
        with st.cond:
            while len(st.stats) < st.N - 1:
                st.cond.wait()
    for j in range(st.N):
        if method in ("B1", "B2"):
            tl.sleep_until(st.arrive[j])
            if j >= next_uniform - 1e-9 and not st.banned[j]:
                add_kf(j)
                next_uniform += step
            continue
        # O methods: wait for the statistics of frame pair (j, j+1) (the last frame has none; reuse the previous pair)
        if j < st.N - 1:
            with st.cond:
                while j not in st.stats:
                    st.cond.wait()
                r = st.stats[j]
            m = r["sum_l1"] / max(r["count"], 1) / 256.0
            sharp[j], thumbs[j] = r["sharp"], r["thumb"]
            S.append(S[-1] + m)
        else:
            sharp[j], thumbs[j] = 0.0, thumbs[j - 1]
        s0 = time.perf_counter()
        if not st.banned[j]:
            if last_S is None:
                if S[j] <= w * T:
                    cand.append(j)
            elif S[j] - last_S >= (1 - w) * T:
                cand.append(j)
        decide = False
        if last_S is None:
            decide = S[j] > w * T or j == st.N - 1
        else:
            decide = S[j] - last_S >= (1 + w) * T or j == st.N - 1
        t_busy += time.perf_counter() - s0
        if decide:
            if not cand and last_S is None and not st.banned[j]:
                cand = [j]
            if cand:
                best = max(cand, key=lambda c: sharp[c])
                add_kf(best)
                last_S = S[best]
                cand = [c for c in cand if c > best and S[c] - last_S >= (1 - w) * T]
            elif last_S is not None and S[j] - last_S >= (1 + w) * T and not st.banned[j]:
                add_kf(j)
                last_S = S[j]
    with st.cond:
        st.select_done = True
        st.cond.notify_all()
    tl.mark("select_done")
    st.select_busy = t_busy


# ============================================================ GPU: incremental COLMAP extraction + matching
def colmap_gpu(st, cm, mdir, img_dir, db, cam, log):
    a, tl = st.a, st.tl
    first = True
    while True:
        with st.cond:
            while True:
                done = st.select_done
                if a.serial and not done:
                    st.cond.wait(0.5)
                    continue
                if st.pending_imgs or st.pending_pairs or done:
                    break
                st.cond.wait(0.5)
            imgs = st.pending_imgs
            st.pending_imgs = []
        for n in imgs:
            HP.link_or_copy(st.paths[st.names.index(n)], os.path.join(img_dir, n))
        if imgs:
            lf = os.path.join(mdir, f"list_{len(st.extracted)}.txt")
            open(lf, "w").write("\n".join(imgs) + "\n")
            ex = ["--database_path", db, "--image_path", img_dir, "--image_list_path", lf]
            ex += cam if first else ["--ImageReader.existing_camera_id", 1]
            s = tl.now()
            with st.db_lock:
                cm.run("feature_extractor", ex + cm.gpu_args("feature_extractor", "Extraction"), log)
            tl.add("GPU SIFT", s, tl.now(), f"{len(imgs)} images")
            first = False
            st.extracted.update(imgs)
        with st.cond:
            ready = [p for p in st.pending_pairs if p[0] in st.extracted and p[1] in st.extracted]
            rs = set(ready)
            st.pending_pairs = [p for p in st.pending_pairs if p not in rs]
            done = st.select_done and not st.pending_imgs and not st.pending_pairs
        if ready:
            pf = os.path.join(mdir, f"pairs_{len(st.matched_imgs)}_{len(ready)}.txt")
            open(pf, "w").write("".join(f"{x} {y}\n" for x, y in ready))
            s = tl.now()
            with st.db_lock:
                cm.run("matches_importer", ["--database_path", db, "--match_list_path", pf, "--match_type", "pairs"]
                       + cm.gpu_args("matches_importer", "Matching"), log)
            tl.add("GPU match", s, tl.now(), f"{len(ready)} pairs")
            with st.cond:
                for x, y in ready:
                    st.matched_imgs.update((x, y))
                st.cond.notify_all()
        if done:
            break
        if not imgs and not ready:           # nothing to do (should not happen; avoids spinning)
            time.sleep(0.05)
    with st.cond:
        st.gpu_done = True
        st.cond.notify_all()
    tl.mark("gpu_done")


# ============================================================ CPU: incremental mapping in batches
def find_model(path):
    if os.path.isfile(os.path.join(path, "images.bin")):
        return path, len(CIO.read_images_bin(os.path.join(path, "images.bin")))
    best, n, _ = CIO.best_model(path)
    return best, n


def mapper(st, cm, mdir, img_dir, db, intr, log):
    a, tl = st.a, st.tl
    chunk = 0
    mapped = 0
    step = max(a.map_step, 1)
    while True:
        with st.cond:
            while True:
                n_ready = len(st.matched_imgs)
                final = st.gpu_done
                if final or (not a.serial and a.mapper == "incremental" and n_ready - mapped >= step):
                    break
                st.cond.wait(0.5)
        snap = os.path.join(mdir, "db_map.db")
        s = tl.now()
        with st.db_lock:
            src = sqlite3.connect(db)
            dst = sqlite3.connect(snap)
            src.backup(dst)
            src.close()
            dst.close()
        out = os.path.join(mdir, f"sparse_c{chunk}")
        if os.path.isdir(out):
            shutil.rmtree(out)
        os.makedirs(out)
        args = ["--database_path", snap, "--image_path", img_dir, "--output_path", out]
        mcmd, pre = ("global_mapper", "GlobalMapper") if a.mapper == "global" else ("mapper", "Mapper")
        if st.model and a.mapper == "incremental":
            args += ["--input_path", st.model]
        if intr:
            args += [f"--{pre}.ba_refine_focal_length", 0, f"--{pre}.ba_refine_principal_point", 0,
                     f"--{pre}.ba_refine_extra_params", 0]
        try:
            cm.run(mcmd, args, log)
            m, n = find_model(out)
            if m:
                st.model = m
        except RuntimeError as e:
            st.errors.append(str(e))
        tl.add("CPU mapping", s, tl.now(), f"batch {chunk}: {n_ready - mapped} images")
        mapped = n_ready
        chunk += 1
        if final:
            break
    tl.mark("map_done")
    st.n_map_chunks = chunk


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="directory produced by hetero_prep flow (contains flow.npz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default="O2", choices=["O2", "O1", "B1", "B2", "none"])
    ap.add_argument("--K", type=int, default=200, help="target number of keyframes (converted to a motion threshold T in pixels)")
    ap.add_argument("--rate", default="offline", choices=["offline", "online"])
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--flow", default="fpga", choices=["fpga", "gpu", "split-static", "split-dynamic"])
    ap.add_argument("--gpu_share", type=float, default=0.5, help="GPU share for split-static")
    ap.add_argument("--serial", action="store_true")
    ap.add_argument("--mapper", default="incremental", choices=["incremental", "global"])
    ap.add_argument("--map_step", type=int, default=50, help="continue mapping after this many new keyframes")
    ap.add_argument("--win", type=float, default=0.5)
    ap.add_argument("--nmin", type=int, default=4)
    ap.add_argument("--loop_k", type=int, default=3)
    ap.add_argument("--loop_thr", type=float, default=0.6)
    ap.add_argument("--seq_overlap", type=int, default=10)
    ap.add_argument("--colmap", default="colmap")
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--no_test", action="store_true", help="do not register test frames (saves time)")
    a = ap.parse_args()

    d, meta = HP.load_flow(a.work)
    test, banned = HP.pick_test_frames(meta["N"], d["lap"])
    st = Stream(a, d, meta, banned, test)
    S_tot = float(np.sum(d["motion"][:st.N - 1]))
    st.T = S_tot / max(a.K - 1, 1)
    mdir = a.out
    if os.path.isdir(mdir):
        shutil.rmtree(mdir)
    os.makedirs(mdir)
    img_dir = os.path.join(mdir, "images")
    os.makedirs(img_dir)
    db = os.path.join(mdir, "database.db")
    intr = meta.get("intrinsics")
    cam = ["--ImageReader.single_camera", 1]
    cam += (["--ImageReader.camera_model", "PINHOLE", "--ImageReader.camera_params", ",".join(f"{x}" for x in intr)]
            if intr else ["--ImageReader.camera_model", "SIMPLE_RADIAL"])
    cm = HP.Colmap(a.colmap)

    use_gpu_flow = a.flow != "fpga" and a.method not in ("B1", "B2")
    if use_gpu_flow:      # frames held by the "camera" (read once, not timed)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(16) as ex:
            st.frames = np.stack(list(ex.map(HP.load_gray, st.paths)))
        import torch
        import gpu_flow as GF
        if torch.cuda.is_available():         # CUDA initialisation and kernel warm-up are not timed
            b = torch.from_numpy(st.frames[:3 * LANES]).cuda()
            GF.pyr_flow_stats(b, b)
            torch.cuda.synchronize()
    # flow device assignment
    fq, gq = queue.Queue(), queue.Queue()
    if a.flow == "fpga":
        queues, assign = [fq], (lambda i: 0)
    elif a.flow == "gpu":
        queues, assign = [gq], (lambda i: 0)
    elif a.flow == "split-static":
        queues = [fq, gq]
        g = a.gpu_share
        assign = lambda i: 1 if int((i + 1) * g) > int(i * g) else 0   # noqa: E731  interleave evenly, giving the GPU a share g
    else:
        queues, assign = [fq], (lambda i: 0)      # shared queue: both FPGA and GPU take from fq
    threads = []
    st.tl = Timeline()                            # timing starts here
    if a.method not in ("B1", "B2"):
        threads.append(threading.Thread(target=camera, args=(st, queues, assign), name="camera"))
        if a.flow in ("fpga", "split-static", "split-dynamic"):
            threads.append(threading.Thread(target=fpga_worker, args=(st, fq), name="fpga"))
        if a.flow == "gpu":
            threads.append(threading.Thread(target=gpu_flow_worker, args=(st, gq), name="gpu_flow"))
        elif a.flow == "split-static":
            threads.append(threading.Thread(target=gpu_flow_worker, args=(st, gq), name="gpu_flow"))
        elif a.flow == "split-dynamic":
            threads.append(threading.Thread(target=gpu_flow_worker, args=(st, fq), name="gpu_flow"))
    else:
        st.tl.marks["capture_end"] = st.arrive[-1]
    if a.method != "none":
        threads.append(threading.Thread(target=selector, args=(st,), name="select"))
        threads.append(threading.Thread(target=colmap_gpu, args=(st, cm, mdir, img_dir, db, cam,
                                                               os.path.join(mdir, "colmap_gpu_log.txt")), name="colmap"))
        threads.append(threading.Thread(target=mapper, args=(st, cm, mdir, img_dir, db, intr,
                                                           os.path.join(mdir, "mapper_log.txt")), name="mapper"))
    else:
        threads.append(threading.Thread(target=selector_none, args=(st,), name="select"))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    st.tl.mark("end")
    with st.cond:
        st.flow_done = True
    flow_end = max([e for lane, s, e, _ in st.tl.ev if lane.startswith(("FPGA", "GPU flow"))], default=0.0)

    tl = st.tl
    res = {"method": a.method, "rate": a.rate, "flow": a.flow, "serial": a.serial, "mapper": a.mapper,
           "K_target": a.K, "T_px": st.T, "N": st.N, "K": len(st.kf), "pairs": len(st.all_pairs), "loops": st.n_loops,
           "marks": tl.marks, "flow_end": flow_end,
           "n_pairs_fpga": sum(1 for v in st.src.values() if v == "fpga"),
           "n_pairs_gpu": sum(1 for v in st.src.values() if v == "gpu"),
           "fpga_lane_busy_s": float(np.sum(getattr(st, "fpga_busy", [0.0]))),
           "gpu_flow_busy_s": getattr(st, "gpu_flow_busy", 0.0),
           "select_busy_s": getattr(st, "select_busy", 0.0), "errors": st.errors}
    busy = {}
    for lane, s, e, _ in tl.ev:
        key = "FPGA" if lane.startswith("FPGA") else lane
        busy[key] = busy.get(key, 0.0) + (e - s)
    res["busy_s"] = busy
    cap = tl.marks.get("capture_end", 0.0)
    res["total_s"] = tl.marks["end"]
    res["after_capture_s"] = tl.marks["end"] - cap
    if a.method != "none":
        res["keyframes"] = st.kf
        res["n_map_chunks"] = getattr(st, "n_map_chunks", 0)
        if st.model:
            res["registered"] = len(CIO.read_images_bin(os.path.join(st.model, "images.bin")))
            gt = os.path.join(meta["seq"], "groundtruth.txt")
            if os.path.isfile(gt):
                gts, gtp = CIO.read_tum_traj(gt)
                cen = CIO.camera_centers(CIO.read_images_bin(os.path.join(st.model, "images.bin")))
                name2ts = {n: float(t) for n, t in zip(st.names, d["ts"][:st.N])}
                res["ate"] = CIO.ate(sorted((name2ts[n], c) for n, c in cen.items() if n in name2ts), gts, gtp)
            if not a.no_test:
                tnames = [st.names[i] for i in test if i < st.N]
                for n in tnames:
                    HP.link_or_copy(st.paths[st.names.index(n)], os.path.join(img_dir, n))
                open(os.path.join(mdir, "test_list.txt"), "w").write("\n".join(tnames) + "\n")
                kfn = [st.names[i] for i in st.kf]
                try:
                    res["test"] = HP.register_test(cm, mdir, db, img_dir, st.model, kfn, tnames, st.kf,
                                                   [i for i in test if i < st.N], cam,
                                                   os.path.join(mdir, "test_log.txt"), intr)
                except RuntimeError as e:
                    res["test"] = {"error": str(e)}
    json.dump(res, open(os.path.join(mdir, "stream_result.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump({"events": tl.ev, "marks": tl.marks}, open(os.path.join(mdir, "timeline.json"), "w", encoding="utf-8"), ensure_ascii=False)
    keep = ("method", "rate", "flow", "serial", "mapper", "K", "pairs", "loops", "flow_end", "total_s",
            "after_capture_s", "registered", "n_pairs_fpga", "n_pairs_gpu", "busy_s", "marks")
    print(json.dumps({k: res.get(k) for k in keep}, ensure_ascii=False, indent=1))
    if "ate" in res:
        print("ATE", res["ate"].get("rmse"))


def selector_none(st):
    """Flow stage only: wait until the statistics of every frame pair have arrived"""
    with st.cond:
        while len(st.stats) < st.N - 1:
            st.cond.wait()
    st.tl.mark("select_done")


if __name__ == "__main__":
    main()

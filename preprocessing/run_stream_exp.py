"""
run_stream_exp.py -- run all "FPGA and GPU working at the same time" experiments in one go (on Windows)

  A. flow work split (offline, flow stage only): fpga / gpu / split-static (by measured throughput ratio) / split-dynamic
  B. pipeline parallelism: O2/B1/B2 x K=200/400 x offline/online (30 fps) x concurrent (stream)/serial
  C. both combined: O2 K=200 offline, flow split-dynamic (the GPU does flow and SIFT at the same time)

  python run_stream_exp.py --work E:\\hetero\\work\\fr3 --out E:\\hetero\\work\\stream --colmap ...\\colmap.exe [--only A]
Results: <out>/<run>/stream_result.json + timeline.json; summary in <out>/summary.json
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def run(a, name, extra):
    out = os.path.join(a.out, name)
    res_p = os.path.join(out, "stream_result.json")
    if a.skip_done and os.path.isfile(res_p):
        print(f"[skip] {name}", flush=True)
        return json.load(open(res_p, encoding="utf-8"))
    cmd = [sys.executable, os.path.join(HERE, "stream_prep.py"), "--work", a.work, "--out", out,
           "--colmap", a.colmap] + [str(x) for x in extra]
    print(f"=== {name}: {' '.join(str(x) for x in extra)}", flush=True)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    os.makedirs(out, exist_ok=True)
    open(os.path.join(a.out, f"{name}.log"), "w", encoding="utf-8").write(r.stdout + "\n" + r.stderr)
    if r.returncode != 0 or not os.path.isfile(res_p):
        print(f"  failed (exit {r.returncode}), see {name}.log", flush=True)
        print(r.stderr[-1500:], flush=True)
        return None
    res = json.load(open(res_p, encoding="utf-8"))
    ate = (res.get("ate") or {}).get("rmse")
    print(f"  {time.perf_counter() - t0:.0f}s | K={res.get('K')} pairs={res.get('pairs')} flow_end={res['flow_end']:.3f}"
          f" total={res['total_s']:.1f} after_capture={res['after_capture_s']:.1f} reg={res.get('registered')}"
          f" ATE={ate}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--colmap", required=True)
    ap.add_argument("--only", default="ABC")
    ap.add_argument("--Ks", nargs="+", type=int, default=[200, 400])
    ap.add_argument("--skip_done", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    summ_p = os.path.join(a.out, "summary.json")
    summ = json.load(open(summ_p, encoding="utf-8")) if os.path.isfile(summ_p) else {}

    def rec(name, extra):
        r = run(a, name, extra)
        if r:
            summ[name] = {k: v for k, v in r.items() if k not in ("keyframes",)}
            json.dump(summ, open(summ_p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        return r

    if "A" in a.only:
        rf = rec("A_flow_fpga", ["--method", "none", "--flow", "fpga"])
        rg = rec("A_flow_gpu", ["--method", "none", "--flow", "gpu"])
        if rf and rg:
            n = rf["N"] - 1
            thr_f, thr_g = n / rf["flow_end"], n / rg["flow_end"]
            share = thr_g / (thr_f + thr_g)
            print(f"  throughput: FPGA {thr_f:.0f} pairs/s, GPU {thr_g:.0f} pairs/s -> GPU share {share:.2f}", flush=True)
            rec("A_flow_split_static", ["--method", "none", "--flow", "split-static", "--gpu_share", f"{share:.3f}"])
        rec("A_flow_split_dynamic", ["--method", "none", "--flow", "split-dynamic"])
        rec("A_flow_gpu_online", ["--method", "none", "--flow", "gpu", "--rate", "online"])
    if "B" in a.only:
        for rate in ("offline", "online"):
            for K in a.Ks:
                for meth in ("O2", "B2", "B1"):
                    if meth == "B1" and K > 200:
                        continue
                    for mode in ("stream", "serial"):
                        extra = ["--method", meth, "--K", K, "--rate", rate, "--map_step", max(25, K // 8), "--no_test"]
                        if mode == "serial":
                            extra.append("--serial")
                        rec(f"B_{rate}_{meth}_K{K}_{mode}", extra)
    if "C" in a.only:
        rec("C_offline_O2_K200_splitdyn", ["--method", "O2", "--K", 200, "--flow", "split-dynamic", "--map_step", 25, "--no_test"])
        rec("C_online_O2_K200_splitdyn", ["--method", "O2", "--K", 200, "--flow", "split-dynamic", "--rate", "online",
                                          "--map_step", 25, "--no_test"])
    print("ALL_STREAM_DONE", flush=True)


if __name__ == "__main__":
    main()

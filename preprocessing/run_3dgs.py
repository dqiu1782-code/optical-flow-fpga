"""
run_3dgs.py -- train and evaluate 3DGS on the COLMAP scenes produced by each method (training itself is standard GPU training, not the focus of this work)

  python run_3dgs.py --work E:\\hetero\\work\\fr3 --gs E:\\hetero\\code\\gs --methods B1 B2 O1 O2 --iters 30000

Per method: train.py --eval (test set = scene/sparse/0/test.txt, the same test frames for all methods) -> render.py --skip_train
-> eval_render.py computes PSNR / SSIM / LPIPS. Results go to <work>/<method>/gs_result.json
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def patch_reader(gs):
    """The official code uses llffhold 8 by default (every 8th image is a test image); changed to use test.txt when the scene has one"""
    p = os.path.join(gs, "scene", "dataset_readers.py")
    s = open(p, encoding="utf-8").read()
    tag = "# [hetero] use test.txt if present"
    if tag in s:
        return
    old = '    if eval:\n        if "360" in path:\n            llffhold = 8\n'
    new = ('    if eval:\n        if "360" in path:\n            llffhold = 8\n'
           f'        {tag}\n'
           '        if os.path.exists(os.path.join(path, "sparse/0", "test.txt")):\n            llffhold = 0\n')
    if old not in s:
        raise SystemExit("dataset_readers.py does not have the expected structure; edit llffhold by hand")
    open(p, "w", encoding="utf-8").write(s.replace(old, new))
    print("patched scene/dataset_readers.py: uses test.txt for the test split when present")


def patch_compat(gs):
    """Compatibility patch for gaussian-splatting main + the older diff-gaussian-rasterization (no antialiasing/depth outputs);
    render.py without torchvision (saves images with PIL). Safe to run repeatedly."""
    import re
    p = os.path.join(gs, "gaussian_renderer", "__init__.py")
    s = open(p, encoding="utf-8").read()
    s2 = re.sub(r",\s*antialiasing=pipe\.antialiasing", "", s)
    s2 = s2.replace("rendered_image, radii, depth_image = rasterizer(", "depth_image = None; rendered_image, radii = rasterizer(")
    if s2 != s:
        open(p, "w", encoding="utf-8").write(s2)
        print("patched gaussian_renderer/__init__.py (older rasterizer interface)")
    p = os.path.join(gs, "render.py")
    s = open(p, encoding="utf-8").read()
    if "import torchvision\n" in s:
        s = s.replace("import torchvision\n", "from PIL import Image as _PILImage; _save_image = lambda t, p: _PILImage.fromarray("
                      "(t.detach().float().mul(255).add(0.5).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy())).save(p)\n")
        s = s.replace("torchvision.utils.save_image(", "_save_image(")
        open(p, "w", encoding="utf-8").write(s)
        print("patched render.py (no torchvision)")


def run(cmd, log, cwd):
    t0 = time.perf_counter()
    with open(log, "a", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        f.flush()
        r = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"{cmd[1]} failed, see {log}")
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--gs", required=True, help="gaussian-splatting source directory")
    ap.add_argument("--methods", nargs="+", default=["B1", "B2", "O1", "O2"])
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--skip_train", action="store_true")
    a = ap.parse_args()
    patch_reader(a.gs)
    patch_compat(a.gs)
    py = sys.executable
    for m in a.methods:
        scene = os.path.join(a.work, m, "scene")
        out = os.path.join(a.work, m, "gs_out")
        log = os.path.join(a.work, m, "gs_log.txt")
        if not os.path.isfile(os.path.join(scene, "sparse", "0", "test.txt")):
            print(f"{m}: no scene (preprocessing failed or not run), skipped")
            continue
        try:
            one(a, m, py, scene, out, log)
        except Exception as e:
            print(f"{m}: failed {e}", flush=True)


def one(a, m, py, scene, out, log):
    res = {"method": m, "iters": a.iters}
    if not a.skip_train:
        res["train_s"] = run([py, "train.py", "-s", scene, "-m", out, "--eval", "--iterations", str(a.iters),
                              "--test_iterations", str(a.iters), "--save_iterations", str(a.iters),
                              "--disable_viewer", "--quiet"], log, a.gs)
    res["render_s"] = run([py, "render.py", "-m", out, "--skip_train", "--iteration", str(a.iters), "--quiet"], log, a.gs)
    rd = os.path.join(out, "test", f"ours_{a.iters}")
    r = subprocess.run([py, os.path.join(HERE, "eval_render.py"), rd], capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-2000:])
    res.update(json.load(open(os.path.join(rd, "metrics_hetero.json"))))
    json.dump(res, open(os.path.join(a.work, m, "gs_result.json"), "w"), indent=2)
    print(f"{m}: PSNR {res['psnr']:.2f}  SSIM {res['ssim']:.4f}  LPIPS {res.get('lpips')}  "
          f"train {res.get('train_s', 0):.0f}s", flush=True)


if __name__ == "__main__":
    main()

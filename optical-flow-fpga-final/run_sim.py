"""
Run all tests in one go -- works on Windows / Mac / Linux

    python run_sim.py           (Windows)
    python3 run_sim.py          (Mac / Linux)

Flow: Python generates test vectors -> iverilog compiles -> vvp simulates -> PASS/FAIL summary
Five test sets x two testbenches (Stage 1 alone + the whole 3-stage lk_top pipeline)
+ 2 multi-lane sets + 5 full pyramid chain sets (tb_lk_pyr.v)
Pyramid only:  python run_sim.py pyr
"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
# (name, lk_golden.py arguments)
CASES = [
    ("smooth", []),
    ("noise", ["--noise"]),
    ("stress", ["--stress"]),                 # solver edge cases: positive/negative saturation, det threshold, full bit width
    ("big", ["--W", "64", "--H", "48"]),
    ("odd", ["--W", "13", "--H", "9"]),       # odd sizes, catches hard-coded coordinates/borders
]
# (testbench name, file list, waveform file name)
BENCHES = [
    ("grad", ["tb/tb_lk_gradient.v", "rtl/lk_gradient.v"], "lk_gradient.vcd", []),
    ("top",  ["tb/tb_lk_top.v", "rtl/lk_top.v", "rtl/lk_gradient.v", "rtl/lk_window.v", "rtl/lk_solve.v"],
     "lk_top.vcd", ["+VCD"]),
]


def need(tool):
    if shutil.which(tool) is None:
        print(f"[error] command not found: {tool}")
        print("  Windows: when installing Icarus Verilog, tick the option that adds it to PATH (Add ... to PATH),")
        print("           then close and reopen the terminal; if it still fails, add C:\\iverilog\\bin to PATH by hand.")
        print("  Mac:     brew install icarus-verilog")
        sys.exit(1)


def main():
    os.chdir(ROOT)
    need("iverilog")
    need("vvp")
    try:
        import numpy  # noqa: F401
    except ImportError:
        print("[error] numpy is missing, run first:  python -m pip install numpy")
        sys.exit(1)

    golden = os.path.join("python", "lk_golden.py")
    results = []
    only_pyr = len(sys.argv) > 1 and sys.argv[1] == "pyr"
    for name, args in ([] if only_pyr else CASES):
        print(f"\n================ test: {name} ================", flush=True)
        subprocess.run([sys.executable, golden, *args], check=True)

        for bname, files, vcd, extra in BENCHES:
            comp = subprocess.run(
                ["iverilog", "-g2012", "-Wall", "-I", "vectors", "-o", "sim.vvp",
                 *[os.path.join(*f.split("/")) for f in files]],
                capture_output=True, text=True)
            if comp.stdout or comp.stderr:
                print(comp.stdout + comp.stderr)
            if comp.returncode != 0:
                print(f"[{bname} compile failed] see the errors above")
                results.append((f"{name}/{bname}", False))
                continue
            sim = subprocess.run(["vvp", "-n", "sim.vvp", "+BUBBLE=30", *extra], capture_output=True, text=True)
            out = sim.stdout + sim.stderr
            print(f"--- {bname} ---")
            print("\n".join(l for l in out.splitlines() if not l.startswith("VCD info") and l.strip()))
            if os.path.exists(vcd):
                shutil.copyfile(vcd, f"wave_{bname}_{name}.vcd")
            results.append((f"{name}/{bname}", "*** PASS ***" in out))

    # ---------------- multi-lane: several image pairs at once ----------------
    MULTI_FILES = ["tb/tb_lk_multi.v", "rtl/lk_multi.v", "rtl/lk_top.v", "rtl/lk_gradient.v",
                   "rtl/lk_window.v", "rtl/lk_solve.v"]
    for name, args in (() if only_pyr else (("multi4", ["--lanes", "4"]), ("multi3odd", ["--lanes", "3", "--W", "15", "--H", "11"]))):
        print(f"\n================ test: {name} ================", flush=True)
        subprocess.run([sys.executable, golden, *args], check=True)
        comp = subprocess.run(["iverilog", "-g2012", "-Wall", "-I", "vectors", "-o", "sim.vvp",
                               *[os.path.join(*f.split("/")) for f in MULTI_FILES]],
                              capture_output=True, text=True)
        if comp.stdout or comp.stderr:
            print(comp.stdout + comp.stderr)
        if comp.returncode != 0:
            results.append((f"{name}/multi", False))
            continue
        sim = subprocess.run(["vvp", "-n", "sim.vvp", "+BUBBLE=30"], capture_output=True, text=True)
        out = sim.stdout + sim.stderr
        print("\n".join(l for l in out.splitlines() if l.strip()))
        results.append((f"{name}/multi", "*** PASS ***" in out))

    # ---------------- pyramid LK: build pyramid + per-level warp/LK + statistics, the whole chain ----------------
    PYR_FILES = ["tb/tb_lk_pyr.v", "rtl/lk_pyramid.v", "rtl/lk_level.v", "rtl/lk_warp.v", "rtl/lk_top.v",
                 "rtl/lk_gradient.v", "rtl/lk_window.v", "rtl/lk_solve.v"]
    pyr = os.path.join("python", "lk_pyramid.py")
    PYR_CASES = [
        ("pyr_shift", ["--W", "64", "--H", "48", "--levels", "3", "--kind", "shift"]),
        ("pyr_vert_odd", ["--W", "61", "--H", "47", "--levels", "3", "--kind", "vert", "--nrb", "5"]),   # odd sizes + vertical clamp (RY=14)
        ("pyr_noise", ["--W", "64", "--H", "48", "--levels", "3", "--kind", "noise", "--det-min", "256"]),
        ("pyr_stress", ["--W", "52", "--H", "36", "--levels", "3", "--kind", "stress", "--det-min", "256"]),
        ("pyr_4lvl", ["--W", "128", "--H", "96", "--levels", "4", "--kind", "shift", "--seed", "5"]),
        # residual clamp +-1 px + 64-row ring buffer (RY=30)
        ("pyr_clamp_nrb6", ["--W", "96", "--H", "80", "--levels", "3", "--kind", "vert", "--du-max", "256", "--nrb", "6"]),
    ]
    for name, args in PYR_CASES:
        print(f"\n================ test: {name} ================", flush=True)
        subprocess.run([sys.executable, pyr, *args, "--vectors", "vectors"], check=True)
        comp = subprocess.run(["iverilog", "-g2012", "-Wall", "-I", "vectors", "-o", "sim_pyr.vvp",
                               *[os.path.join(*f.split("/")) for f in PYR_FILES]],
                              capture_output=True, text=True)
        if comp.stdout or comp.stderr:
            print(comp.stdout + comp.stderr)
        if comp.returncode != 0:
            results.append((f"{name}/pyr", False))
            continue
        sim = subprocess.run(["vvp", "-n", "sim_pyr.vvp", "+BUBBLE=20"], capture_output=True, text=True)
        out = sim.stdout + sim.stderr
        print("\n".join(l for l in out.splitlines() if l.strip()))
        results.append((f"{name}/pyr", "*** PASS ***" in out))

    # restore vectors/ to the default set, for comparing with the waveform
    subprocess.run([sys.executable, golden], check=True, stdout=subprocess.DEVNULL)

    print("\n================ summary ================")
    for name, ok in results:
        print(f"  {name:<14} {'PASS' if ok else 'FAIL'}")
    if all(ok for _, ok in results):
        print("\nAll passed. Waveform:  gtkwave wave_top_smooth.vcd")
    else:
        print("\nSome tests failed -- see the full output above")
        sys.exit(1)


if __name__ == "__main__":
    main()

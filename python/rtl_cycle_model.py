"""
rtl_cycle_model.py -- rtl/lk_gradient.v translated to Python register by register, cycle by cycle.

Why this file exists:
  The most common Verilog bug is not a wrong formula but misaligned timing -- e.g. a line buffer read one cycle early/late,
  or a border check off by one row. This model follows the RTL structure exactly (same registers, same line buffers,
  same "all registers update together on the rising edge" semantics), runs with random bubbles + two consecutive frames,
  and compares pixel by pixel with the lk_golden.py reference. It does not replace a real Verilog simulation
  (it cannot check Verilog syntax or bit widths), but it validates the architecture before iverilog is installed.

Run:  python3 rtl_cycle_model.py        (run lk_golden.py first to generate vectors/)
"""
import os
import random

X = None   # models an uninitialised Verilog 'x'


def load_hex(path, bits):
    vals = [int(line, 16) for line in open(path, encoding="utf-8") if line.strip()]
    if bits == 9:   # 9-bit two's complement -> Python signed int
        vals = [v - 512 if v >= 256 else v for v in vals]
    return vals


def run(W, H, f0, f1, nframes=2, bubble=0.3, seed=1):
    rng = random.Random(seed)
    # ---- registers / memories (reset values; line buffers have no reset = X) ----
    lb0_r1, lb0_r2, lb1_r1 = [X] * W, [X] * W, [X] * W
    state, col, row, left_r1 = "STREAM", 0, 0, 0
    outputs, done_count = [], 0

    i, frame = 0, 0
    for _cycle in range(100 * W * H * nframes):
        # ---- testbench stimulus ----
        feeding = frame < nframes
        in_valid = feeding and rng.random() >= bubble
        p0 = f0[i] if feeding else 0
        p1 = f1[i] if feeding else 0

        # ---- combinational logic (all inputs are pre-edge values) ----
        in_ready = state == "STREAM"
        fire = in_valid and in_ready
        up = lb0_r1[col] if col < W else X
        up2 = lb0_r2[col] if col < W else X
        upr = 0 if col == W - 1 else lb0_r1[col + 1]
        up_i1 = lb1_r1[col]
        border = row == 1 or col == 0 or col == W - 1

        def sub(a, b):
            return X if (a is X or b is X) else a - b

        ix_raw, iy_raw, it_raw = sub(upr, left_r1), sub(p0, up2), sub(up_i1, up)

        # ---- rising edge: compute all new register values (non-blocking semantics: compute first, update together) ----
        n_state, n_col, n_row, n_left = state, col, row, left_r1
        out = None
        done = False
        wr = None
        if state == "STREAM":
            if fire:
                wr = (col, up, p0, p1)
                n_left = up
                if row != 0:
                    vals = (0, 0, 0) if border else (ix_raw, iy_raw, it_raw)
                    out = (row - 1, col) + vals
                if col == W - 1:
                    n_col = 0
                    if row == H - 1:
                        n_row, n_state = 0, "FLUSH"
                    else:
                        n_row = row + 1
                else:
                    n_col = col + 1
        else:  # FLUSH
            out = (H - 1, col, 0, 0, 0)
            if col == W - 1:
                n_col, n_state, done = 0, "STREAM", True
            else:
                n_col = col + 1

        # ---- update together ----
        if wr:
            c, a, b, d = wr
            lb0_r2[c], lb0_r1[c], lb1_r1[c] = a, b, d
        state, col, row, left_r1 = n_state, n_col, n_row, n_left
        if out:
            outputs.append(out)
        if done:
            done_count += 1

        # ---- testbench handshake counters ----
        if fire:
            i += 1
            if i == W * H:
                i, frame = 0, frame + 1
        if done_count == nframes:
            return outputs
    raise RuntimeError("timeout")


def main():
    vec = os.path.join(os.path.dirname(__file__), "..", "vectors")
    params = open(os.path.join(vec, "params.vh"), encoding="utf-8").read()
    W = int(params.split("IMG_W =")[1].split(";")[0])
    H = int(params.split("IMG_H =")[1].split(";")[0])
    f0, f1 = load_hex(f"{vec}/frame0.hex", 8), load_hex(f"{vec}/frame1.hex", 8)
    g = [load_hex(f"{vec}/golden_{k}.hex", 9) for k in ("ix", "iy", "it")]

    for bubble in (0.0, 0.3, 0.7):
        for seed in range(3):
            outs = run(W, H, f0, f1, nframes=2, bubble=bubble, seed=seed)
            errs = 0
            for n, (y, x, ix, iy, it) in enumerate(outs):
                idx = y * W + x
                if idx != n % (W * H) or (ix, iy, it) != (g[0][idx], g[1][idx], g[2][idx]):
                    errs += 1
                    if errs <= 5:
                        print(f"  mismatch #{n} (y={y},x={x}): got {(ix, iy, it)}, "
                              f"expected {(g[0][idx], g[1][idx], g[2][idx])}")
            ok = errs == 0 and len(outs) == 2 * W * H
            print(f"bubble={bubble:.0%} seed={seed}: {len(outs)} outputs, {errs} errors -> "
                  f"{'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()

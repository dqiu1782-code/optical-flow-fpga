"""
MPI Sintel dataset tools: read images, convert to grey, read ground-truth flow (.flo), compute error (EPE).

Layout (after unzipping MPI-Sintel-training_images.zip + MPI-Sintel-training_extras.zip):
    <root>/training/clean/<sequence>/frame_0001.png ...
    <root>/training/flow/<sequence>/frame_0001.flo ...   (GT flow from frame_000k to frame_000k+1)
"""
import os
import glob
import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


def to_gray_u8(rgb):
    """Integer luma Y = (77R + 150G + 29B) >> 8 -- every platform (CPU/GPU/FPGA) uses the same grey image."""
    rgb = rgb.astype(np.uint32)
    return ((77 * rgb[..., 0] + 150 * rgb[..., 1] + 29 * rgb[..., 2]) >> 8).astype(np.uint8)


def read_gray(path):
    if Image is None:
        raise SystemExit("Pillow missing, run:  python -m pip install pillow")
    return to_gray_u8(np.asarray(Image.open(path).convert("RGB")))


def read_flo(path):
    with open(path, "rb") as f:
        tag = np.fromfile(f, np.float32, 1)[0]
        if abs(tag - 202021.25) > 1e-3:
            raise ValueError(f"{path} is not a .flo file")
        w, h = np.fromfile(f, np.int32, 2)
        data = np.fromfile(f, np.float32, 2 * w * h).reshape(h, w, 2)
    return data[..., 0], data[..., 1]


def list_pairs(root, pass_name="clean"):
    """Return [(seq, img0, img1, flo), ...] sorted by sequence and frame number."""
    pairs = []
    img_root = os.path.join(root, "training", pass_name)
    flo_root = os.path.join(root, "training", "flow")
    for seq in sorted(os.listdir(img_root)):
        frames = sorted(glob.glob(os.path.join(img_root, seq, "frame_*.png")))
        for a, b in zip(frames[:-1], frames[1:]):
            flo = os.path.join(flo_root, seq, os.path.basename(a).replace(".png", ".flo"))
            if os.path.exists(flo):
                pairs.append((seq, a, b, flo))
    return pairs


def epe_stats(u, v, gu, gv, flag, border=2):
    """End-point error EPE = sqrt((u-gu)^2 + (v-gv)^2) in pixels, excluding a border of `border` pixels.
    Returns: EPE over all pixels, EPE over flag=1 pixels, flag=1 fraction, EPE over small-motion pixels (GT < 1 px)."""
    sl = (slice(border, -border), slice(border, -border))
    u, v, gu, gv, flag = (a[sl] for a in (u, v, gu, gv, flag))
    e = np.sqrt((u - gu) ** 2 + (v - gv) ** 2)
    small = np.sqrt(gu ** 2 + gv ** 2) < 1.0
    return {
        "epe_all": float(e.mean()),
        "epe_flag": float(e[flag].mean()) if flag.any() else float("nan"),
        "flag_frac": float(flag.mean()),
        "epe_small_flag": float(e[flag & small].mean()) if (flag & small).any() else float("nan"),
    }

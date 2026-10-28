"""eval_render.py <render dir (with renders/ and gt/)> -- PSNR / SSIM (Gaussian window 11, sigma=1.5) / LPIPS (VGG, when it can be loaded)"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)[None]


def ssim(x, y, ws=11, sigma=1.5):
    g = torch.exp(-((torch.arange(ws) - ws // 2) ** 2).float() / (2 * sigma ** 2))
    g = (g / g.sum())
    w = (g[:, None] @ g[None, :])[None, None].repeat(3, 1, 1, 1).to(x)
    mu_x = F.conv2d(x, w, padding=ws // 2, groups=3)
    mu_y = F.conv2d(y, w, padding=ws // 2, groups=3)
    sxx = F.conv2d(x * x, w, padding=ws // 2, groups=3) - mu_x ** 2
    syy = F.conv2d(y * y, w, padding=ws // 2, groups=3) - mu_y ** 2
    sxy = F.conv2d(x * y, w, padding=ws // 2, groups=3) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return m.mean().item()


def main():
    d = sys.argv[1]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        import lpips
        net = lpips.LPIPS(net="vgg", verbose=False).to(dev)
    except Exception as e:  # lpips not installed or VGG weights not downloadable
        print("LPIPS unavailable:", e)
        net = None
    names = sorted(os.listdir(os.path.join(d, "renders")))
    ps, ss, ls = [], [], []
    per = {}
    for n in names:
        x = load(os.path.join(d, "renders", n)).to(dev)
        y = load(os.path.join(d, "gt", n)).to(dev)
        mse = torch.mean((x - y) ** 2).item()
        p = 10 * np.log10(1.0 / max(mse, 1e-12))
        s = ssim(x, y)
        l = net(x * 2 - 1, y * 2 - 1).item() if net is not None else None
        ps.append(p)
        ss.append(s)
        if l is not None:
            ls.append(l)
        per[n] = {"psnr": p, "ssim": s, "lpips": l}
    res = {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)), "lpips": float(np.mean(ls)) if ls else None,
           "n_test": len(names), "per_image": per}
    json.dump(res, open(os.path.join(d, "metrics_hetero.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_image"}))


if __name__ == "__main__":
    main()

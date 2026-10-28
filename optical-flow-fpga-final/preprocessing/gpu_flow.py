"""
gpu_flow.py -- GPU (PyTorch, float32) version of pyramid LK + per-frame statistics, same algorithm as the FPGA:
2x2 average downsampling (same +2 rounding), previous-level flow x2 as the guess, vertical guess clamp +-RY, bilinear warp,
5x5 window LK, per-level residual clamp +-DU px, det threshold; statistics converted to the same units as the FPGA registers (Q8).
Also computes the sharpness sum(dx²+dy²) with the same definition as lk_sharp, and the level-4 thumbnail.
Values are float, so they are not bit-identical to the FPGA fixed-point results (differences of order 1/256 px), which is fine for keyframe selection.
"""
import torch
import torch.nn.functional as F

LEVELS, RY, DU, DET_MIN = 5, 30, 1.0, 65536.0


def _lk(I0, I1, det_min):
    Ix = torch.zeros_like(I0)
    Iy = torch.zeros_like(I0)
    It = torch.zeros_like(I0)
    Ix[:, 1:-1, 1:-1] = I0[:, 1:-1, 2:] - I0[:, 1:-1, :-2]
    Iy[:, 1:-1, 1:-1] = I0[:, 2:, 1:-1] - I0[:, :-2, 1:-1]
    It[:, 1:-1, 1:-1] = I1[:, 1:-1, 1:-1] - I0[:, 1:-1, 1:-1]
    P = torch.stack([Ix * Ix, Ix * Iy, Iy * Iy, Ix * It, Iy * It], dim=1)
    S = F.pad(F.avg_pool2d(P, kernel_size=5, stride=1) * 25.0, (2, 2, 2, 2))
    Sxx, Sxy, Syy, Sxt, Syt = S.unbind(1)
    det = Sxx * Syy - Sxy * Sxy
    flag = det > det_min
    safe = torch.where(flag, det, torch.ones_like(det))
    u = torch.where(flag, -2.0 * (Syy * Sxt - Sxy * Syt) / safe, torch.zeros_like(det))
    v = torch.where(flag, -2.0 * (Sxx * Syt - Sxy * Sxt) / safe, torch.zeros_like(det))
    return u, v, flag


def _down(x):
    B, H, W = x.shape
    h, w = H // 2, W // 2
    s = x[:, 0:2 * h:2, 0:2 * w:2] + x[:, 0:2 * h:2, 1:2 * w:2] + x[:, 1:2 * h:2, 0:2 * w:2] + x[:, 1:2 * h:2, 1:2 * w:2]
    return torch.floor((s + 2) / 4)


@torch.no_grad()
def pyr_flow_stats(I0, I1, levels=LEVELS, ry=RY, du=DU, det_min=DET_MIN):
    """I0, I1: (B,H,W) uint8 tensors (already on the GPU). Returns per-pair statistics (Python lists)"""
    P0, P1 = [I0.float()], [I1.float()]
    for _ in range(levels - 1):
        P0.append(_down(P0[-1]))
        P1.append(_down(P1[-1]))
    u = v = None
    for l in range(levels - 1, -1, -1):
        a, b = P0[l], P1[l]
        B, H, W = a.shape
        if u is None:
            gu = torch.zeros_like(a)
            gv = torch.zeros_like(a)
        else:
            Hc, Wc = u.shape[1:]
            ys = torch.clamp(torch.arange(H, device=a.device) // 2, max=Hc - 1)
            xs = torch.clamp(torch.arange(W, device=a.device) // 2, max=Wc - 1)
            gu = (2 * u)[:, ys][:, :, xs]
            gv = torch.clamp((2 * v)[:, ys][:, :, xs], -ry, ry)
        yy, xx = torch.meshgrid(torch.arange(H, device=a.device, dtype=a.dtype),
                                torch.arange(W, device=a.device, dtype=a.dtype), indexing="ij")
        px = (xx + gu).clamp(0, W - 1 - 1 / 256)
        py = (yy + gv).clamp(0, H - 1 - 1 / 256)
        grid = torch.stack([px / (W - 1) * 2 - 1, py / (H - 1) * 2 - 1], -1)
        bw = F.grid_sample(b[:, None], grid, mode="bilinear", padding_mode="border", align_corners=True)[:, 0]
        ddu, ddv, flag = _lk(a, bw, det_min)
        u = gu + ddu.clamp(-du, du)
        v = gv + ddv.clamp(-du, du)
    f = flag.float()
    l1 = (u.abs() + v.abs()) * f
    a0 = P0[0].double()
    dx = a0[:, 1:, 1:] - a0[:, 1:, :-1]
    dy = a0[:, 1:, 1:] - a0[:, :-1, 1:]
    sharp = (dx * dx + dy * dy).sum((1, 2))
    res = torch.stack([f.sum((1, 2)).double(), (l1.sum((1, 2)) * 256).double(), ((u * f).sum((1, 2)) * 256).double(),
                       ((v * f).sum((1, 2)) * 256).double(), sharp], 1).cpu().numpy()
    thumbs = P0[-1].to(torch.uint8).cpu().numpy()
    return [{"count": int(r[0]), "sum_l1": float(r[1]), "sum_u": float(r[2]), "sum_v": float(r[3]),
             "sharp": float(r[4]), "thumb": thumbs[k]} for k, r in enumerate(res)]

# -*- coding: utf-8 -*-
"""生成形状匹配 vs NCC 对比结果图（全景+放大，带图例/误差标注）"""
import cv2
import numpy as np
import os

TPL = r"D:\Users\Administrator\Desktop\weights222\object_region_full_res.jpg"
SMP = r"D:\Users\Administrator\Desktop\weights222\corresponding_sample_region_full_res.jpg"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shape_match_test_output")

tpl = cv2.imread(TPL, 0); smp = cv2.imread(SMP, 0)
FH, FW = tpl.shape[:2]
f = 2
tg = cv2.resize(tpl, (FW // f, FH // f), interpolation=cv2.INTER_AREA).astype(np.float32)
sg = cv2.resize(smp, (FW // f, FH // f), interpolation=cv2.INTER_AREA).astype(np.float32)


def hp(g, s=8):
    k = int(s * 3) | 1
    return g - cv2.GaussianBlur(g, (k, k), s)


def ofield(img, pct, double):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, 3); gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, 3)
    mag = np.sqrt(gx * gx + gy * gy)
    m = mag >= np.percentile(mag, pct)
    G = np.zeros(img.shape, np.complex128)
    if double:
        G[m] = ((gx[m] ** 2 - gy[m] ** 2) + 2j * gx[m] * gy[m]) / (mag[m] ** 2)
    else:
        G[m] = (gx[m] + 1j * gy[m]) / mag[m]
    return G, int(m.sum())


def best_shift(tg_hp, sg_hp, pct_t, pct_s, double):
    Gt, nt = ofield(tg_hp, pct_t, double)
    Gs, _ = ofield(sg_hp, pct_s, double)
    C = np.fft.ifft2(np.fft.fft2(Gt) * np.conj(np.fft.fft2(Gs))).real / nt
    py, px = np.unravel_index(np.argmax(C), C.shape)
    if py > C.shape[0] / 2: py -= C.shape[0]
    if px > C.shape[1] / 2: px -= C.shape[1]
    return int(px), int(py), float(C.max())


# NCC 基准 H（1/2 分辨率，与 shape_match_test.py 一致）
T, S, H2 = 160, 96, 60
half = T // 2
pts, us = [], []
for ty in range(H2 + half, sg.shape[0] - H2 - half, S):
    for tx in range(H2 + half, sg.shape[1] - H2 - half, S):
        r = cv2.matchTemplate(sg[ty - H2 - half:ty + H2 + half, tx - H2 - half:tx + H2 + half],
                              tg[ty - half:ty + half, tx - half:tx + half], cv2.TM_CCOEFF_NORMED)
        _, s, _, loc = cv2.minMaxLoc(r)
        if s > 0.6:
            pts.append((tx, ty)); us.append((loc[0] - H2, loc[1] - H2))
P = np.array(pts, float); U = np.array(us, float)
Hm, _ = cv2.findHomography(P.astype(np.float32), (P + U).astype(np.float32), cv2.USAC_MAGSAC, 3.0)
from scipy.spatial import cKDTree
for _ in range(3):
    _, nb = cKDTree(P).query(P, k=12)
    dev = np.linalg.norm(U - np.median(U[nb], axis=1), axis=1)
    g = dev < max(6, 4 * (np.median(dev) + 1e-6))
    if g.all(): break
    P, U = P[g], U[g]
    Hm, _ = cv2.findHomography(P.astype(np.float32), (P + U).astype(np.float32), cv2.USAC_MAGSAC, 3.0)
S4 = np.diag([2, 2, 1]).astype(np.float64)
H_full = S4 @ Hm.astype(np.float64) @ np.diag([.5, .5, 1])
S2 = np.diag([.5, .5, 1]).astype(np.float64)

# 两个形状匹配位姿（平移相似变换，角度最优均≈0°）
dx_std, dy_std, sc_std = best_shift(hp(tg), hp(sg), 97, 97, double=False)
dx_dbl, dy_dbl, sc_dbl = best_shift(hp(tg), hp(sg), 97, 97, double=True)
print(f"standard: d=({dx_std},{dy_std}) score={sc_std:.4f} | double-angle: d=({dx_dbl},{dy_dbl}) score={sc_dbl:.4f}")


def sim_M(dx, dy):
    """半分辨率坐标系下的平移相似变换（角度≈0）"""
    M = np.eye(2, 3); M[:, 2] = [dx, dy]
    return M


M_std = sim_M(dx_std, dy_std)
M_dbl = sim_M(dx_dbl, dy_dbl)
sel = np.random.default_rng(7).choice(len(P), min(800, len(P)), replace=False)
Pt = P[sel] * 2.0; Ps = (P + U)[sel] * 2.0
def err(M_half):
    # 半分辨率平移 ×2 = 全分辨率误差
    return float(np.median(np.linalg.norm(Pt @ M_half[:, :2].T + M_half[:, 2] * 2 - Ps, axis=1)))
e_std, e_dbl = err(M_std), err(M_dbl)
e_h = float(np.median(np.linalg.norm(
    cv2.perspectiveTransform(Pt.reshape(-1, 1, 2).astype(np.float32), H_full).reshape(-1, 2) - Ps, axis=1)))
print(f"median err: std={e_std:.1f}px double={e_dbl:.1f}px H={e_h:.2f}px")

# ---- 绘图 ----
edges = cv2.Canny(cv2.resize(tpl, (FW // 2, FH // 2)), 50, 150)
col = np.zeros((FH // 2, FW // 2, 3), np.uint8); col[..., 1] = edges
def warp_edges(M, mode):
    if mode == "affine":
        return cv2.warpAffine(col, M, (FW // 2, FH // 2))
    return cv2.warpPerspective(col, M, (FW // 2, FH // 2))
w_std = warp_edges(M_std, "affine")[..., 1] > 0
w_dbl = warp_edges(M_dbl, "affine")[..., 1] > 0
invS2 = np.diag([2, 2, 1]).astype(np.float64)
w_h = warp_edges(S2 @ H_full @ invS2, "persp")[..., 1] > 0   # 半分辨率单应: 线性=I, 平移折半

base = cv2.cvtColor(sg, cv2.COLOR_GRAY2BGR)
ov = base.copy()
ov[w_std] = (0, 0, 255)      # 红=标准形状匹配
ov[w_dbl] = (0, 255, 0)      # 绿=抗反极性
ov[w_h] = (255, 0, 255)      # 品红=NCC+H

scale = 1800 / (FW // 2)
full = cv2.resize(ov, (1800, int(FH // 2 * scale)))
zx, zy = 520, 640; zw, zh = 1500, 1000
zoom = cv2.resize(ov[zy:zy + zh, zx:zx + zw], (1800, int(zh * 1800 / zw)))
zoom2 = cv2.resize(ov[zy + 40:zy + 40 + 700, zx + 3200:zx + 3200 + 1400], (1500, int(700 * 1500 / 1400)))
zoomrow = np.full((zoom.shape[0], 60, 3), 20, np.uint8)
zoom = np.hstack([zoom, zoomrow, cv2.resize(zoom2, (int(zoom2.shape[1] * zoom.shape[0] / zoom2.shape[0]), zoom.shape[0]))]) if False else zoom


def legend_bar(w):
    bar = np.full((110, w, 3), 18, np.uint8)
    items = [((0, 0, 255), f"Shape match (polarity): median err {e_std:.0f}px - FAILED"),
             ((0, 255, 0), f"Shape match (polarity-insensitive): {e_dbl:.0f}px"),
             ((255, 0, 255), f"NCC + MAGSAC H: {e_h:.2f}px")]
    x = 20
    for c, t in items:
        cv2.rectangle(bar, (x, 40), (x + 46, 70), c, -1)
        cv2.putText(bar, t, (x + 58, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (235, 235, 235), 2)
        x += 20 + 58 + int(len(t) * 16.5) + 40
    return bar


fig = np.vstack([legend_bar(1800), full])
ok, buf = cv2.imencode(".png", fig); buf.tofile(os.path.join(OUT, "compare_full.png"))
figz = np.vstack([legend_bar(1800), zoom])
ok, buf = cv2.imencode(".png", figz); buf.tofile(os.path.join(OUT, "compare_zoom.png"))
print("saved compare_full.png / compare_zoom.png")

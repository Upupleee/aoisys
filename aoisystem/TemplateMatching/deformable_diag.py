# -*- coding: utf-8 -*-
"""独立验证：图像空间中刚性 vs TPS 的实际对齐质量"""
import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
import sys

tpl = cv2.imread(r'D:\Users\Administrator\Desktop\weights222\object_region_full_res.jpg', 0)
smp = cv2.imread(r'D:\Users\Administrator\Desktop\weights222\corresponding_sample_region_full_res.jpg', 0)
FH, FW = tpl.shape
f = 2
wh, ww = FH // f, FW // f
tw = cv2.resize(tpl, (ww, wh))
sw = cv2.resize(smp, (ww, wh))

T, D, S = 160, 96, 60
half = T // 2
pts, u, sc = [], [], []
for ty in range(half + S, wh - half - S, D):
    for tx in range(half + S, ww - half - S, D):
        r = cv2.matchTemplate(sw[ty - S - half:ty + S + half, tx - S - half:tx + S + half],
                              tw[ty - half:ty + half, tx - half:tx + half],
                              cv2.TM_CCOEFF_NORMED)
        _, s, _, loc = cv2.minMaxLoc(r)
        if s > 0.6:
            pts.append((tx, ty))
            u.append((loc[0] - S, loc[1] - S))
            sc.append(s)
pts = np.array(pts, float); u = np.array(u, float); sc = np.array(sc)
P, U = pts[sc > 0.6], u[sc > 0.6]
H, _ = cv2.findHomography(P.astype(np.float32), (P + U).astype(np.float32), cv2.USAC_MAGSAC, 3.0)
tree = cKDTree(P)
_, nb = tree.query(P, k=12)
med = np.median(U[nb], axis=1)
dev = np.linalg.norm(U - med, axis=1)
g = dev < max(6, 4 * np.median(dev) + 1e-6)
P, U = P[g], U[g]
H, _ = cv2.findHomography(P.astype(np.float32), (P + U).astype(np.float32), cv2.USAC_MAGSAC, 3.0)
norm = np.array([FW, FH], float)
tps = RBFInterpolator((P * f) / norm, U * f, kernel='thin_plate_spline', smoothing=0.1, neighbors=64)
back = tps((P * f) / norm)
print(f"identity check: max|tps(P)-U*f| = {np.abs(back - U*f).max():.2f}px full-res")
print(f"U*f stats: median={np.median(np.linalg.norm(U*f,axis=1)):.1f} max={np.linalg.norm(U*f,axis=1).max():.1f}")
print(f"P*f/norm range: x[{(P*f/norm)[:,0].min():.3f},{(P*f/norm)[:,0].max():.3f}] y[{(P*f/norm)[:,1].min():.3f},{(P*f/norm)[:,1].max():.3f}]")

rng = np.random.default_rng(0)
idx = rng.choice(len(P), 25, replace=False)
e_rig_l, e_tps_l = [], []
print(" point |  err_rigid  err_TPS")
for i in idx:
    pw = P[i]                                          # work-res 模板坐标
    pred_w = cv2.perspectiveTransform(np.float32([[pw]]), H).reshape(2)  # work-res 采样坐标
    pred = pred_w * f                                  # full-res
    up = tps(((pw * f) / norm).reshape(1, 2))[0]       # TPS 位移 (full-res px)
    T2 = 80
    patch = tw[int(P[i, 1]) - T2:int(P[i, 1]) + T2, int(P[i, 0]) - T2:int(P[i, 0]) + T2]
    # 在预测位置附近小窗内搜真实位置
    best = {}
    for name, pos in (("rigid", pred), ("tps", pred + up)):
        cx, cy = int(pos[0] / f), int(pos[1] / f)
        cy = min(max(cy, 100), wh - 100); cx = min(max(cx, 100), ww - 100)
        reg = sw[cy - 100:cy + 100, cx - 100:cx + 100]
        _, s, _, loc = cv2.minMaxLoc(cv2.matchTemplate(reg, patch, cv2.TM_CCOEFF_NORMED))
        act = np.array([cx - 100 + loc[0] + T2, cy - 100 + loc[1] + T2], float)  # work-res
        best[name] = np.linalg.norm(act - pos / f)
    e_rig_l.append(best["rigid"]); e_tps_l.append(best["tps"])
    print(f" {int(P[i,0]):5d},{int(P[i,1]):5d} |   {best['rigid']:7.2f}  {best['tps']:7.2f}")
print(f"\nmedian err: rigid={np.median(e_rig_l):.2f}px  TPS={np.median(e_tps_l):.2f}px (工作分辨率)")
print(f"TPS 更优的点数: {sum(t < r for r, t in zip(e_rig_l, e_tps_l))}/{len(idx)}")

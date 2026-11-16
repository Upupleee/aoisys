# -*- coding: utf-8 -*-
"""重绘 halcon_vs_ncc 对比图（修复 R/F 缩放 bug，不改任何算法结果）"""
import cv2
import numpy as np
import os

TPL = r"D:\Users\Administrator\Desktop\weights222\object_region_full_res.jpg"
SMP = r"D:\Users\Administrator\Desktop\weights222\corresponding_sample_region_full_res.jpg"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shape_match_test_output")

# 全量跑得到的 Halcon 级位姿（全分辨率坐标: q = R(a)(p-c)+c+T, c=模板中心）
A_FIN_DEG = 0.149
T_FIN = np.array([2.34, 7.84])

F = 2
tpl = cv2.imread(TPL, 0)
smp = cv2.imread(SMP, 0)
FH, FW = tpl.shape
tg = cv2.resize(tpl, (FW // F, FH // F), interpolation=cv2.INTER_AREA)
sg = cv2.resize(smp, (FW // F, FH // F), interpolation=cv2.INTER_AREA)
Hh, Wh = sg.shape

# --- NCC+MAGSAC (参照) ---
pts, us = [], []
half = 80
for ty in range(half + 60, Hh - half - 60, 96):
    for tx in range(half + 60, Wh - half - 60, 96):
        r = cv2.matchTemplate(sg[ty - 60 - half:ty + 60 + half, tx - 60 - half:tx + 60 + half],
                              tg[ty - half:ty + half, tx - half:tx + half], cv2.TM_CCOEFF_NORMED)
        _, s, _, loc = cv2.minMaxLoc(r)
        if s > 0.6:
            pts.append((tx, ty)); us.append((loc[0] - 60, loc[1] - 60))
Pn = np.array(pts, float); Un = np.array(us, float)
H2, _ = cv2.findHomography(Pn.astype(np.float32), (Pn + Un).astype(np.float32),
                           cv2.USAC_MAGSAC, 3.0)
from scipy.spatial import cKDTree
for _ in range(3):
    _, nb = cKDTree(Pn).query(Pn, k=12)
    dev = np.linalg.norm(Un - np.median(Un[nb], axis=1), axis=1)
    g = dev < max(6, 4 * (np.median(dev) + 1e-6))
    if g.all():
        break
    Pn, Un = Pn[g], Un[g]
    H2, _ = cv2.findHomography(Pn.astype(np.float32), (Pn + Un).astype(np.float32),
                               cv2.USAC_MAGSAC, 3.0)
S4 = np.diag([F, F, 1]).astype(np.float64)
invS = np.diag([1 / F, 1 / F, 1]).astype(np.float64)
# 画图用: 半分辨率模板边缘 -> 半分辨率采样画布, 就是 H2 本身(半->半)
# (S·H2·S⁻¹ 是全分辨率单应, 平移会×2, 不能用于半分辨率画布!)
H_half = H2.astype(np.float64)

# --- Halcon 级位姿 -> 半分辨率仿射 (线性=R, 平移=(I-R)c_half+T_half) ---
a_ = np.radians(A_FIN_DEG)
R = np.array([[np.cos(a_), -np.sin(a_)], [np.sin(a_), np.cos(a_)]])
c_half = np.array([FW / (2.0 * F), FH / (2.0 * F)])
T_half = T_FIN / F
M_fin = np.eye(2, 3)
M_fin[:, :2] = R
M_fin[:, 2] = (np.eye(2) - R) @ c_half + T_half

edges = cv2.Canny(tg, 50, 150)
col = np.zeros((Hh, Wh, 3), np.uint8)
col[..., 1] = edges
w_lsq = cv2.warpAffine(col, M_fin, (Wh, Hh))[..., 1] > 0
w_ncc = cv2.warpPerspective(col, H_half, (Wh, Hh))[..., 1] > 0

base = cv2.cvtColor(sg, cv2.COLOR_GRAY2BGR)
ov = base.copy()
ov[w_lsq] = (255, 255, 0)   # 青 = Halcon级
ov[w_ncc] = (255, 0, 255)   # 品红 = NCC
ok, buf = cv2.imencode(".png", ov)
buf.tofile(os.path.join(OUT, "halcon_vs_ncc_full.png"))
zy, zx = 640, 520
zoom = ov[zy:zy + 1000, zx:zx + 1500]
ok, buf = cv2.imencode(".png", cv2.resize(zoom, (1800, 1200)))
buf.tofile(os.path.join(OUT, "halcon_vs_ncc_zoom.png"))
print("saved halcon_vs_ncc_full.png / halcon_vs_ncc_zoom.png")

# -*- coding: utf-8 -*-
"""506s4e pair1：边缘域 H0 → 矫正 → ECC → 验收 完整链路验证。"""
import cv2
import numpy as np
import os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from val_2652s2g import load_down, T, accept, warp_by_H
from retest_phase_corr import fm_measure, fm_refine_translation, compose_H
from make_overlay_heatmap import local_match, overlay, heatmap_points

OUT = os.path.join(HERE, "val_506s4e_output")

tpl = load_down(r"D:\AOI\aoisys\506s4e\template\template_1.tif", 8)
smp = load_down(r"D:\AOI\aoisys\506s4e\sample\test_imgs1.tif", 8)
h_t, w_t = tpl.shape
Wt, Ht = w_t, h_t
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(32, 32))


def prep_edge(img):
    x = clahe.apply(img)
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    e = cv2.magnitude(gx, gy)
    return np.clip(e, 0, np.percentile(e, 99)).astype(np.uint8)


t_e = prep_edge(255 - tpl)
s_e = prep_edge(smp)
Hc, Wc = max(h_t, smp.shape[0]), max(w_t, smp.shape[1])
otx, oty = (Wc - w_t) // 2, (Hc - h_t) // 2
osx, osy = (Wc - smp.shape[1]) // 2, (Hc - smp.shape[0]) // 2


def pad(im, ox, oy):
    p = np.zeros((Hc, Wc), np.uint8)
    p[oy:oy + im.shape[0], ox:ox + im.shape[1]] = im
    return p


th_raw, ls_raw, resp, _, _ = fm_measure(pad(t_e, otx, oty), pad(s_e, osx, osy))
th, s_ = -th_raw, float(np.exp(ls_raw))
tx, ty, _ = fm_refine_translation(pad(t_e, otx, oty), pad(s_e, osx, osy), th, s_)
H0_pad = compose_H(th, s_, -tx, -ty, Wc, Hc)
H0 = T(-osx, -osy) @ H0_pad @ T(otx, oty)
print(f"边缘域 H0: θ={th:.4f} s={s_:.4f} t=({-tx:.1f},{-ty:.1f}) (工作分辨率 px)")

# 多候选平移搜索：pc 基础上按单元周期网格扩展（重复板平移面多峰）
# 简化：pc / pc±256 / pc±512 / t=0 共 13 个候选，各做局部 ZNCC 验证取最高
cands = [(0.0, 0.0)]
for d in (128, 256, 384, 512):
    cands += [(tx - d, -ty), (tx + d, -ty), (tx, -ty - d), (tx, -ty + d)]
gx = np.arange(64, w_t - 64, 128, dtype=np.float64)
gy = np.arange(64, h_t - 64, 128, dtype=np.float64)
Xg, Yg = np.meshgrid(gx, gy)
grid = np.stack([Xg.ravel(), Yg.ravel()], -1)
t_a, s_a = clahe.apply(tpl), clahe.apply(smp)

best = (None, -2.0, None)
for k, (cx, cy) in enumerate(cands):
    H0 = T(-osx, -osy) @ compose_H(th, s_, cx, cy, Wc, Hc) @ T(otx, oty)
    wr = np.c_[grid, np.ones(len(grid))] @ H0.T
    mapped = wr[:, :2] / wr[:, 2:3]
    found, sc = local_match(t_a, s_a, grid, mapped, patch=64, search=20)
    okv = np.isfinite(found[:, 0]) & (sc >= 0.4)
    score = float(np.median(sc[okv])) if okv.mean() > 0.1 else -2.0
    if score > best[1]:
        best = (H0, score, (cx, cy, float(okv.mean())))
    if k < 6 or score > 0.5:
        r = (np.linalg.norm(mapped[okv] - found[okv], axis=1).mean() if okv.sum() else -1)
        print(f"  候选 t=({cx:.0f},{cy:.0f}): 内点率 {okv.mean():.1%} zncc_med {score:.3f} 平均残差 {r:.1f}px")
H0, score, info = best
print(f"最佳平移候选: t={info[:2]} 内点率 {info[2]:.1%} zncc_med {score:.3f}")

# 矫正 + ECC 精化
smp_al0 = warp_by_H(smp, H0, Wt, Ht)
Hm = np.eye(3)
for f in (4, 2, 1):
    tw = cv2.resize(tpl, (Wt // f, Ht // f), interpolation=cv2.INTER_AREA)
    sw = cv2.resize(smp_al0, (Wt // f, Ht // f), interpolation=cv2.INTER_AREA)
    S = np.diag([1.0 / f, 1.0 / f, 1.0])
    wmat = S @ Hm @ np.diag([f, f, 1.0])
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-7)
    try:
        _, wmat = cv2.findTransformECC(tw.astype(np.float32), sw.astype(np.float32),
                                       wmat.astype(np.float32).copy(),
                                       cv2.MOTION_HOMOGRAPHY, crit, None, 5)
        Hm = np.diag([f, f, 1.0]) @ wmat.astype(np.float64) @ np.diag([1.0 / f, 1.0 / f, 1.0])
        print(f"  ECC f={f}: 收敛")
    except cv2.error as e:
        print(f"  ECC f={f}: 失败")
        break
H_rel = Hm
ok1, m1, pts1, err1 = accept(tpl, smp_al0, H_rel)
print(f"最终验收: {'通过' if ok1 else '未通过'} {m1}")
if len(err1):
    overlay(tpl, warp_by_H(smp_al0, H_rel, Wt, Ht),
            os.path.join(OUT, "p1_ov_edge_ecc.png"), "Edge-domain FM + ECC (506s4e p1)")
    heatmap_points(tpl, pts1, err1, os.path.join(OUT, "p1_hm_edge_ecc.png"),
                   f"边缘域先验→ECC 残差分布 ({len(pts1)} 验收点)")

# -*- coding: utf-8 -*-
"""506s4e 修正实验：极性对齐(模板取反) + 采样 CLAHE 增强 → FM 矫正 → ECC → NCC 全链。
归因修正：三批模板均来自图纸转换（统计一致）；506 失败根因是采样侧低对比(std=20)
+ 8% 缩放差叠加，而非"渲染 vs 实拍"。本实验验证采样增强能否打通链路。"""
import cv2
import numpy as np
import os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from val_2652s2g import load_down, coarse_prior, accept, warp_by_H, T
from retest_phase_corr import calibrate, fm_full
from classical_feature_test import refine, mag_h
import synthetic_tps_validation as stv
from make_overlay_heatmap import local_match, overlay, heatmap_points

OUT = os.path.join(HERE, "val_506s4e_output")
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(32, 32))

tpl = load_down(r"D:\AOI\aoisys\506s4e\template\template_1.tif", 8)
smp_raw = load_down(r"D:\AOI\aoisys\506s4e\sample\test_imgs1.tif", 8)
tpl = 255 - tpl                       # 极性对齐：模板白底黑线 → 黑底白线
smp = clahe.apply(smp_raw)            # 采样对比度增强
print(f"增强前 std={smp_raw.std():.1f}, 增强后 std={smp.std():.1f}")
Wt, Ht = tpl.shape[1], tpl.shape[0]

# ---- FM 矫正（增强图上）----
H0, diag = coarse_prior(tpl, smp)
print(f"先验: {diag}")
if H0 is None:
    sys.exit("先验失败")
ok0, m0, *_ = accept(tpl, smp, H0)
print(f"先验验收: {m0}")

# ---- 矫正 + ECC ----
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
        print(f"ECC f={f}: 收敛")
    except cv2.error:
        print(f"ECC f={f}: 失败")
        break
H_rel = Hm
ok1, m1, pts1, err1 = accept(tpl, smp_al0, H_rel)
print(f"ECC 后验收: {'通过' if ok1 else '未通过'} {m1}")

# ---- NCC 三遍（对齐对）----
smp_al = warp_by_H(smp_al0, H_rel, Wt, Ht)
Hn, pts, disp = None, None, None
Hm2 = np.eye(3)
keep_last = None
for srch, patch, ncc_min in [(stv.SEARCH_R, stv.PATCH, stv.NCC_MIN),
                             (48, stv.PATCH, stv.NCC_MIN),
                             (stv.SMALL_SEARCH, stv.SMALL_PATCH, stv.SMALL_NCC)]:
    pts, disp, sc = stv.match_grid(tpl, smp_al, H0=Hm2, search_r=srch, patch=patch)
    keep = stv.consistency_filter(pts, disp, sc, ncc_min=ncc_min)
    print(f"  pass: 内点 {int(keep.sum())}/{len(pts)}")
    if keep.sum() < 30:
        pts = None
        break
    pts, disp = pts[keep], disp[keep]
    Hm2, _ = stv.fit_homography(pts, disp)
if pts is not None:
    pred = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32),
                                    Hm2).reshape(-1, 2)
    err = np.linalg.norm(pts + disp - pred, axis=1)
    print(f"NCC 三遍(对齐后): {len(pts)} 点, 残差 med {np.median(err):.2f}px p95 {np.percentile(err, 95):.2f}px")
    ok2, m2, *_ = accept(tpl, smp_al, Hm2)
    print(f"NCC-H 验收: {'通过' if ok2 else '未通过'} {m2}")
    overlay(tpl, warp_by_H(smp_al, Hm2, Wt, Ht), os.path.join(OUT, "p1_ov_clahe_chain.png"),
            "506 p1: invert+CLAHE+FM->ECC->NCC (aligned pair)")
    heatmap_points(tpl, pts, err, os.path.join(OUT, "p1_hm_clahe_chain.png"),
                   f"NCC 残差分布 ({len(pts)} 点, 增强链路)")
    # TPS
    rbf, lam, hold_rmse, tscale, hold_h = stv.fit_tps(pts, disp, Hm2)
    if rbf is not None:
        print(f"TPS 门控: 启用 (留出 TPS {hold_rmse:.2f} vs H {hold_h:.2f}px)")
    else:
        print("TPS 门控: 禁用")
else:
    print("NCC 三遍失败（内点不足）")

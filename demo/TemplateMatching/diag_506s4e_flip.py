# -*- coding: utf-8 -*-
"""506s4e 镜像假设验证：用户目检发现采样单元与模板左右相反（疑 Bottom 视图）。
实验：采样水平翻转 + 模板极性对齐(取反) → FM 矫正 → ECC → NCC 全链。"""
import cv2
import numpy as np
import os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from val_2652s2g import load_down, coarse_prior, accept, warp_by_H
import synthetic_tps_validation as stv
from make_overlay_heatmap import overlay, heatmap_points

OUT = os.path.join(HERE, "val_506s4e_output")

tpl = load_down(r"D:\AOI\aoisys\506s4e\template\template_1.tif", 8)
smp_raw = load_down(r"D:\AOI\aoisys\506s4e\sample\test_imgs1.tif", 8)
tpl = 255 - tpl                      # 极性对齐（模板白底黑线 → 黑底白线）
smp = cv2.flip(smp_raw, 1)           # 水平镜像（Bottom 视图假设）
Wt, Ht = tpl.shape[1], tpl.shape[0]
print(f"载入完成: tpl {Wt}x{Ht}, smp 已水平翻转")

# ---- FM 矫正（灰度域直接试——镜像后灰度结构可能可相关）----
H0, diag = coarse_prior(tpl, smp)
print(f"先验: {diag}")
ok0, m0, *_ = accept(tpl, smp, H0)
print(f"先验验收: {'通过' if ok0 else '未通过'} {m0}")

# ---- 矫正 → ECC ----
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

# ---- NCC 三遍 ----
smp_al = warp_by_H(smp_al0, H_rel, Wt, Ht)
Hm2 = np.eye(3)
pts = disp = None
for srch, patch, ncc_min in [(stv.SEARCH_R, stv.PATCH, stv.NCC_MIN),
                             (48, stv.PATCH, stv.NCC_MIN),
                             (stv.SMALL_SEARCH, stv.SMALL_PATCH, stv.SMALL_NCC)]:
    p_, d_, sc_ = stv.match_grid(tpl, smp_al, H0=Hm2, search_r=srch, patch=patch)
    keep = stv.consistency_filter(p_, d_, sc_, ncc_min=ncc_min)
    print(f"  NCC pass: 内点 {int(keep.sum())}/{len(p_)}")
    if keep.sum() < 30:
        pts = None
        break
    pts, disp = p_[keep], d_[keep]
    Hm2, _ = stv.fit_homography(pts, disp)
if pts is not None:
    pred = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32),
                                    Hm2).reshape(-1, 2)
    err = np.linalg.norm(pts + disp - pred, axis=1)
    print(f"NCC 三遍: {len(pts)} 点, med {np.median(err):.2f}px p95 {np.percentile(err, 95):.2f}px")
    ok2, m2, *_ = accept(tpl, smp_al, Hm2)
    print(f"NCC-H 验收: {'通过' if ok2 else '未通过'} {m2}")
    rbf, lam, hold_rmse, tscale, hold_h = stv.fit_tps(pts, disp, Hm2)
    print(f"TPS 门控: {'启用' if rbf is not None else '禁用'}"
          + (f" (留出 TPS {hold_rmse:.2f} vs H {hold_h:.2f}px)" if rbf is not None else ""))
    # 最终贴图（对齐后翻转回原始方向展示：把对齐结果再镜像回来与原采样方向一致）
    smp_al_final = cv2.flip(smp_al, 1)
    overlay(tpl, cv2.flip(warp_by_H(smp_al, Hm2, Wt, Ht), 1),
            os.path.join(OUT, "p1_ov_flip.png"),
            "506 p1: MIRRORED sample + FM->ECC->NCC (display flipped back)")
    heatmap_points(tpl, pts, err, os.path.join(OUT, "p1_hm_flip.png"),
                   f"镜像假设 NCC 残差 ({len(pts)} 点)")
    print(f"-> {OUT}/p1_ov_flip.png, p1_hm_flip.png")
else:
    print("NCC 三遍失败")
    overlay(tpl, smp_al, os.path.join(OUT, "p1_ov_flip_fail.png"),
            "MIRRORED sample: alignment FAILED")

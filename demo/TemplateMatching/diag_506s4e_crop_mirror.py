# -*- coding: utf-8 -*-
"""506s4e：裁掉模板/采样边框 → 采样水平镜像 → FM矫正→ECC→NCC 全链验证。"""
import cv2
import numpy as np
import os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from val_2652s2g import load_down, coarse_prior, accept, warp_by_H
import synthetic_tps_validation as stv
from make_overlay_heatmap import overlay, heatmap_points

OUT = os.path.join(HERE, "val_506s4e_output")
CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(32, 32))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def content_bbox(img, min_len=200, margin=0.02):
    """自动检测内容区 bbox：梯度幅值密度投影 + 阈值 + 最长连续区间。"""
    x = CLAHE.apply(img) if img.std() < 60 else img
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, 3)
    mag = cv2.magnitude(gx, gy)
    mag = np.clip(mag, 0, np.percentile(mag, 99))
    binm = (mag > np.percentile(mag, 80)).astype(np.float32)

    def longest_range(dens, total):
        d = np.convolve(dens, np.ones(max(5, total // 100)) / max(5, total // 100), mode="same")
        th = 0.12 * d.max()
        on = np.where(d > th)[0]
        if len(on) == 0:
            return 0, total
        m = int(margin * total)
        return max(0, on[0] - m), min(total, on[-1] + m)

    col_d = binm.mean(axis=0)
    row_d = binm.mean(axis=1)
    x0, x1 = longest_range(col_d, img.shape[1])
    y0, y1 = longest_range(row_d, img.shape[0])
    return (x0, y0, x1, y1)


def main():
    tpl_raw = load_down(r"D:\AOI\aoisys\506s4e\template\template_1.tif", 8)
    smp_raw = load_down(r"D:\AOI\aoisys\506s4e\sample\test_imgs1.tif", 8)
    tpl_inv = 255 - tpl_raw               # 极性对齐（黑底白线）

    # ---- 自动裁边框 ----
    bx_t = content_bbox(tpl_inv)
    bx_s = content_bbox(smp_raw)
    log(f"模板内容区 bbox={bx_t} (原 {tpl_raw.shape[1]}x{tpl_raw.shape[0]})")
    log(f"采样内容区 bbox={bx_s} (原 {smp_raw.shape[1]}x{smp_raw.shape[0]})")
    tpl = tpl_inv[bx_t[1]:bx_t[3], bx_t[0]:bx_t[2]]
    smp = cv2.flip(smp_raw[bx_s[1]:bx_s[3], bx_s[0]:bx_s[2]], 1)   # 裁剪后水平镜像
    Wt, Ht = tpl.shape[1], tpl.shape[0]
    log(f"裁剪+镜像后: tpl {Wt}x{Ht}, smp {smp.shape[1]}x{smp.shape[0]}")

    # ---- 全链 ----
    H0, diag = coarse_prior(tpl, smp)
    log(f"先验: {diag}")
    if H0 is None:
        log("FAIL-SAFE：先验失败（自标定未过）")
        return
    ok0, m0, *_ = accept(tpl, smp, H0)
    log(f"先验验收: {'通过' if ok0 else '未通过'} {m0}")

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
            log(f"ECC f={f}: 收敛")
        except cv2.error:
            log(f"ECC f={f}: 失败")
            break
    H_rel = Hm
    ok1, m1, pts1, err1 = accept(tpl, smp_al0, H_rel)
    log(f"ECC 后验收: {'通过' if ok1 else '未通过'} {m1}")

    smp_al = warp_by_H(smp_al0, H_rel, Wt, Ht)
    Hm2 = np.eye(3)
    pts = disp = None
    for srch, patch, ncc_min in [(stv.SEARCH_R, stv.PATCH, stv.NCC_MIN),
                                 (48, stv.PATCH, stv.NCC_MIN),
                                 (stv.SMALL_SEARCH, stv.SMALL_PATCH, stv.SMALL_NCC)]:
        p_, d_, sc_ = stv.match_grid(tpl, smp_al, H0=Hm2, search_r=srch, patch=patch)
        keep = stv.consistency_filter(p_, d_, sc_, ncc_min=ncc_min)
        log(f"  NCC pass: 内点 {int(keep.sum())}/{len(p_)}")
        if keep.sum() < 30:
            pts = None
            break
        pts, disp = p_[keep], d_[keep]
        Hm2, _ = stv.fit_homography(pts, disp)
    if pts is None:
        log("NCC 三遍失败")
        overlay(tpl, smp_al, os.path.join(OUT, "p1_ov_cropmirror_fail.png"),
                "crop+mirror: FAILED after ECC")
        return
    pred = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32),
                                    Hm2).reshape(-1, 2)
    err = np.linalg.norm(pts + disp - pred, axis=1)
    log(f"NCC 三遍: {len(pts)} 点, med {np.median(err):.2f}px p95 {np.percentile(err, 95):.2f}px")
    ok2, m2, *_ = accept(tpl, smp_al, Hm2)
    log(f"NCC-H 验收: {'通过' if ok2 else '未通过'} {m2}")
    rbf, lam, hold_rmse, tscale, hold_h = stv.fit_tps(pts, disp, Hm2)
    log(f"TPS 门控: {'启用' if rbf is not None else '禁用'}"
        + (f" (留出 TPS {hold_rmse:.2f} vs H {hold_h:.2f}px)" if rbf is not None else ""))

    # 贴图（镜像回原方向展示）
    ov_src = warp_by_H(smp_al, Hm2, Wt, Ht)
    overlay(tpl, ov_src, os.path.join(OUT, "p1_ov_cropmirror.png"),
            "506 p1: crop-border + mirror sample + FM->ECC->NCC")
    heatmap_points(tpl, pts, err, os.path.join(OUT, "p1_hm_cropmirror.png"),
                   f"裁边框+镜像 NCC 残差 ({len(pts)} 点)")

    # 原始方向对照贴图（采样翻转回未镜像方向，便于与原图对照）
    ov_fl = warp_by_H(cv2.flip(smp_al, 1), cv2.flip(Hm2, 1) if False else Hm2, Wt, Ht)
    log("完成 → p1_ov_cropmirror.png / p1_hm_cropmirror.png")


if __name__ == "__main__":
    main()

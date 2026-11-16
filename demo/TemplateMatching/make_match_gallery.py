# -*- coding: utf-8 -*-
"""
匹配结果图集：统一风格输出各算法"模板-采样匹配结果"图（连线图/对齐叠加图）。
已有（直接复用）：SIFT/ORB、ALIKED+LG、SP+LG、LoFTR。
本脚本生成：
  1. ncc_matches.png         —— NCC pass1 网格控制点连线（绿=一致性内点，红=被滤除）
  2. fm_ecc_matches.png      —— FM→ECC 全局链路：H 映射网格 + 局部 ZNCC 验证连线
  3. overlay_h.png           —— 纯 H 翘曲叠加（模板=红通道，翘曲采样=绿通道，对齐=黄）
  4. overlay_h_tps.png       —— H+TPS 翘曲叠加
  5. overlay_flow.png        —— Farnebäck 光流翘曲叠加
  6. match_gallery.png       —— 全部方法缩略总览
输出：match_gallery_output/
"""
import cv2
import numpy as np
import os, time, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import synthetic_tps_validation as stv
from retest_phase_corr import calibrate, compose_H

OUT = os.path.join(HERE, "match_gallery_output")
os.makedirs(OUT, exist_ok=True)
DOWNSCALE = 0.5
PANEL_W = 1500  # 单侧面板宽


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def draw_matches(tpl, smp, pts_t, pts_s, path, colors=None, title=""):
    """左模板右采样，画匹配连线。colors: 每点 (B,G,R) 或 None=全绿。"""
    s = PANEL_W / tpl.shape[1]
    h = int(tpl.shape[0] * s)
    gap = 12
    canvas = np.full((h, PANEL_W * 2 + gap, 3), 255, np.uint8)
    canvas[:, :PANEL_W] = cv2.cvtColor(cv2.resize(tpl, (PANEL_W, h)), cv2.COLOR_GRAY2BGR)
    canvas[:, PANEL_W + gap:] = cv2.cvtColor(cv2.resize(smp, (PANEL_W, h)), cv2.COLOR_GRAY2BGR)
    n = len(pts_t)
    idx = np.linspace(0, n - 1, min(n, 900)).astype(int) if n > 900 else np.arange(n)
    for i in idx:
        c = (0, 170, 0) if colors is None else tuple(int(v) for v in colors[i])
        p1 = (int(pts_t[i, 0] * s), int(pts_t[i, 1] * s))
        p2 = (int(pts_s[i, 0] * s) + PANEL_W + gap, int(pts_s[i, 1] * s))
        cv2.line(canvas, p1, p2, c, 1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 2, c, -1)
        cv2.circle(canvas, p2, 2, c, -1)
    if title:
        cv2.putText(canvas, title, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(path, canvas)
    log(f"  -> {os.path.basename(path)} ({n} 对, 显示 {len(idx)})")


def overlay(tpl, warped, path, title=""):
    """双色叠加：黑底，红=模板, 绿=翘曲采样，对齐处呈黄；附中心放大窗。"""
    h, w = tpl.shape
    ov = np.zeros((h, w, 3), np.uint8)
    ov[..., 2] = tpl
    ov[..., 1] = warped
    s = PANEL_W / w
    full = cv2.resize(ov, (PANEL_W, int(h * s)))
    cz = int(PANEL_W * 0.22)
    cy, cx = h // 2, w // 2
    zoom = cv2.resize(ov[cy - cz:cy + cz, cx - cz:cx + cz], (int(2 * cz * s), int(2 * cz * s)),
                      interpolation=cv2.INTER_NEAREST)
    zh, zw = zoom.shape[:2]
    canvas = np.full((max(full.shape[0], zh + 16), PANEL_W, 3), 255, np.uint8)
    canvas[:full.shape[0]] = full
    canvas[8:8 + zh, PANEL_W - zw - 8:PANEL_W - 8] = zoom
    cv2.rectangle(canvas, (PANEL_W - zw - 8, 8), (PANEL_W - 8, 8 + zh), (0, 0, 0), 1)
    if title:
        cv2.putText(canvas, title, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, title, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(path, canvas)
    log(f"  -> {os.path.basename(path)}")


def local_verify(tpl, smp, pts_s, patch=96, search=14, min_score=0.5):
    """在采样图 pts_s 位置做局部 ZNCC 验证，返回 (ok掩码, 分数)。"""
    h, w = smp.shape
    pad = search + 2
    smp_pad = cv2.copyMakeBorder(smp, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    half = patch // 2
    ok = np.zeros(len(pts_s), bool)
    score = np.zeros(len(pts_s))
    for i, (sx, sy) in enumerate(pts_s):
        sx, sy = int(round(sx)), int(round(sy))
        x0, y0 = sx - half, sy - half
        if x0 < 0 or y0 < 0 or x0 + patch > w or y0 + patch > h:
            continue
        patch_t = tpl[y0:y0 + patch, x0:x0 + patch]
        if patch_t.std() < 4:
            continue
        win = smp_pad[sy - half:sy - half + patch + 2 * pad, sx - half:sx - half + patch + 2 * pad]
        r = cv2.matchTemplate(win, patch_t, cv2.TM_CCOEFF_NORMED)
        score[i] = r.max()
        ok[i] = score[i] >= min_score
    return ok, score


def main():
    log("读取图像…")
    tpl0 = cv2.imread(stv.TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
    smp0 = cv2.imread(r"D:\Users\Administrator\Desktop\weights222\corresponding_sample_region_full_res.jpg",
                      cv2.IMREAD_GRAYSCALE)
    Wf, Hf = int(tpl0.shape[1] * DOWNSCALE), int(tpl0.shape[0] * DOWNSCALE)
    tpl = cv2.resize(tpl0, (Wf, Hf), interpolation=cv2.INTER_AREA)
    smp = cv2.resize(smp0, (Wf, Hf), interpolation=cv2.INTER_AREA)

    # ---------- NCC pass1 ----------
    log("== NCC pass1 ==")
    pts, disp, sc = stv.match_grid(tpl, smp)
    keep = stv.consistency_filter(pts, disp, sc)
    H_ncc, _ = stv.fit_homography(pts[keep], disp[keep])
    colors = np.full((len(pts), 3), (0, 170, 0), np.uint8)
    colors[~keep] = (0, 0, 255)
    draw_matches(tpl, smp, pts, pts + disp, os.path.join(OUT, "ncc_matches.png"), colors,
                 f"NCC pass1: {int(keep.sum())}/{len(pts)} grid matches")
    H_tps_ctx = (pts[keep], disp[keep])

    # ---------- FM -> ECC ----------
    log("== FM -> ECC 全局链路 ==")
    t0 = time.time()
    cal = calibrate(tpl, Wf, Hf, "PCB")
    H_fm, th, s_, t_, _, _, zncc = (None, None, None, None, None, None, None)
    ok_pcb = cal[0]
    H_ecc = None
    if ok_pcb:
        # 重跑 fm_full + ECC（复用 retest_phase_corr 逻辑的最小子集）
        from retest_phase_corr import fm_full
        H_fm, th, s_, t_, _, _, zncc = fm_full(tpl, smp, cal, Wf, Hf)
        Hm = H_fm.copy()
        cc = -1.0
        for f in (4, 2, 1):
            tw = cv2.resize(tpl, (Wf // f, Hf // f), interpolation=cv2.INTER_AREA)
            sw = cv2.resize(smp, (Wf // f, Hf // f), interpolation=cv2.INTER_AREA)
            S = np.diag([1.0 / f, 1.0 / f, 1.0])
            wmat = S @ Hm @ np.diag([f, f, 1.0])
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-7)
            try:
                cc, wmat = cv2.findTransformECC(tw.astype(np.float32), sw.astype(np.float32),
                                                wmat.astype(np.float32).copy(), cv2.MOTION_HOMOGRAPHY, crit, None, 5)
            except cv2.error:
                break
            Hm = np.diag([f, f, 1.0]) @ wmat.astype(np.float64) @ np.diag([1.0 / f, 1.0 / f, 1.0])
        H_ecc = Hm
        log(f"FM->ECC 完成 cc={cc:.4f}, {time.time() - t0:.0f}s")
    if H_ecc is not None:
        step = 192
        gx = np.arange(step // 2, Wf - step // 2, step, dtype=np.float64)
        gy = np.arange(step // 2, Hf - step // 2, step, dtype=np.float64)
        X, Y = np.meshgrid(gx, gy)
        grid = np.stack([X.ravel(), Y.ravel()], axis=-1)
        wr = np.c_[grid, np.ones(len(grid))] @ H_ecc.T
        mapped = wr[:, :2] / wr[:, 2:3]
        ok, _ = local_verify(tpl, smp, mapped)
        colors = np.full((len(grid), 3), (0, 170, 0), np.uint8)
        colors[~ok] = (0, 0, 255)
        draw_matches(tpl, smp, grid, mapped, os.path.join(OUT, "fm_ecc_matches.png"), colors,
                     f"FM->ECC global: {int(ok.sum())}/{len(grid)} verified grid pts")

    # ---------- 形变场翘曲叠加 ----------
    log("== 形变场翘曲叠加 ==")
    Xf, Yf = np.meshgrid(np.arange(Wf, dtype=np.float32), np.arange(Hf, dtype=np.float32))
    wpts = np.c_[Xf.ravel(), Yf.ravel(), np.ones(Xf.size, np.float32)] @ H_ncc.T.astype(np.float32)
    hfull = (wpts[:, :2] / wpts[:, 2:3]).astype(np.float32)
    warp_h = cv2.remap(smp, hfull[:, 0].reshape(Hf, Wf), hfull[:, 1].reshape(Hf, Wf),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    overlay(tpl, warp_h, os.path.join(OUT, "overlay_h.png"), "Warp by NCC homography (R=tpl G=sample)")

    # TPS（与 retest_farneback_tps 相同的 80/20 训练划分，仅训练点拟合）
    from scipy.interpolate import RBFInterpolator
    kp, dp = H_tps_ctx
    rngs = np.random.default_rng(0)
    idxs = rngs.permutation(len(kp))
    ntr = int(0.8 * len(kp))
    rbf, lam, _, tscale, _ = stv.fit_tps(kp[idxs[:ntr]], dp[idxs[:ntr]], H_ncc)
    if rbf is not None:
        xs8 = np.arange(0, Wf, 8, dtype=np.float64)
        ys8 = np.arange(0, Hf, 8, dtype=np.float64)
        X8, Y8 = np.meshgrid(xs8, ys8)
        g8 = np.stack([X8.ravel(), Y8.ravel()], axis=-1)
        rr8 = rbf(g8 / tscale).astype(np.float32)
        ny8, nx8 = len(ys8), len(xs8)
        mx = hfull[:, 0].reshape(Hf, Wf) + cv2.resize(rr8[:, 0].reshape(ny8, nx8), (Wf, Hf),
                                                      interpolation=cv2.INTER_LINEAR)
        my = hfull[:, 1].reshape(Hf, Wf) + cv2.resize(rr8[:, 1].reshape(ny8, nx8), (Wf, Hf),
                                                      interpolation=cv2.INTER_LINEAR)
        warp_tps = cv2.remap(smp, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        overlay(tpl, warp_tps, os.path.join(OUT, "overlay_h_tps.png"), "Warp by NCC + TPS (R=tpl G=sample)")
    else:
        log("  TPS 被门控禁用，跳过")

    # Farnebäck
    log("计算 Farnebäck 前向光流…")
    t0 = time.time()
    flow = cv2.calcOpticalFlowFarneback(tpl, smp, None, 0.5, 6, 41, 10, 7, 1.5,
                                        cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    log(f"光流 {time.time() - t0:.0f}s")
    warp_fl = cv2.remap(smp, Xf + flow[..., 0].astype(np.float32), Yf + flow[..., 1].astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    overlay(tpl, warp_fl, os.path.join(OUT, "overlay_flow.png"), "Warp by Farneback flow (R=tpl G=sample)")

    # ---------- 总览拼图 ----------
    log("== 总览拼图 ==")
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    items = [
        ("NCC pass1 (主链路)", os.path.join(OUT, "ncc_matches.png")),
        ("ALIKED+LightGlue", os.path.join(HERE, "dl_match_output", "aliked_lightglue_matches.png")),
        ("SuperPoint+LightGlue", os.path.join(HERE, "gapfill_superpoint_output", "superpoint_matches.png")),
        ("LoFTR", os.path.join(HERE, "gapfill_loftr_output", "loftr_matches.png")),
        ("SIFT", os.path.join(HERE, "classical_feature_output", "sift_matches.png")),
        ("ORB", os.path.join(HERE, "classical_feature_output", "orb_matches.png")),
        ("FM->ECC 全局链路", os.path.join(OUT, "fm_ecc_matches.png")),
        ("纯 H 翘曲叠加", os.path.join(OUT, "overlay_h.png")),
        ("H+TPS 翘曲叠加", os.path.join(OUT, "overlay_h_tps.png")),
    ]
    items = [(t, p) for t, p in items if os.path.exists(p)]
    n = len(items)
    ncol = 2
    nrow = (n + 1) // 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 5.2 * nrow))
    axes = np.atleast_2d(axes)
    for k, (title, p) in enumerate(items):
        ax = axes[k // ncol, k % ncol]
        ax.imshow(mpimg.imread(p))
        ax.set_title(title, fontsize=13)
        ax.axis("off")
    for k in range(n, nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    fig.suptitle("各算法 模板-采样 匹配结果（weights222 @0.5x）", fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "match_gallery.png"), dpi=110)
    plt.close(fig)
    log("完成 -> match_gallery_output/match_gallery.png")


if __name__ == "__main__":
    main()

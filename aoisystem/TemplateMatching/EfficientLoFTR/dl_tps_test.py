# -*- coding: utf-8 -*-
"""
深度学习匹配 + 可变形校正验证：ALIKED+LightGlue → MAGSAC H → 残差 TPS → 稠密 remap
与 deformable_tps_test.py（NCC 版）同一套评估/可视化，仅匹配前端换成 DL。
"""
import cv2
import numpy as np
import torch
from lightglue import LightGlue, ALIKED
from lightglue.utils import rbd
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import uniform_filter
import os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "EfficientLoFTR"))

TEMPLATE_PATH = r"D:\Users\Administrator\Desktop\weights222\object_region_full_res.jpg"
SAMPLE_PATH   = r"D:\Users\Administrator\Desktop\weights222\corresponding_sample_region_full_res.jpg"
OUT_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dl_tps_test_output")

MATCH_WIDTH    = 2400
VIS_WIDTH      = 3000
MAX_KEYPOINTS  = 4096
TPS_NEIGHBORS  = 64
LAMBDA_SWEEP   = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
HOLDOUT_FRAC   = 0.25
CROP_SIZE      = 700
N_CROPS        = 4


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"设备: {dev}")

    log("加载全分辨率图像 ...")
    tpl_bgr = cv2.imread(TEMPLATE_PATH)
    smp_bgr = cv2.imread(SAMPLE_PATH)
    FH, FW = tpl_bgr.shape[:2]
    log(f"模板/采样图尺寸: {FW}x{FH}")

    s = MATCH_WIDTH / FW
    mw, mh = MATCH_WIDTH, int(round(FH * s))
    tpl_s = cv2.resize(tpl_bgr, (mw, mh), interpolation=cv2.INTER_AREA)
    smp_s = cv2.resize(smp_bgr, (mw, mh), interpolation=cv2.INTER_AREA)

    def to_tensor(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(rgb).permute(2, 0, 1)[None].to(dev)

    log("ALIKED 特征提取 ...")
    extractor = ALIKED(max_num_keypoints=MAX_KEYPOINTS, detection_threshold=0.1).eval().to(dev)
    matcher = LightGlue(features="aliked").eval().to(dev)
    with torch.inference_mode():
        t0 = time.time()
        feats0 = extractor.extract(to_tensor(tpl_s))
        feats1 = extractor.extract(to_tensor(smp_s))
        log(f"关键点: {feats0['keypoints'].shape[1]} / {feats1['keypoints'].shape[1]} "
            f"({time.time()-t0:.1f}s)")
        log("LightGlue 匹配 ...")
        pred = matcher({'image0': feats0, 'image1': feats1})
        feats0, feats1, pred = [rbd(x) for x in [feats0, feats1, pred]]
        m = pred['matches']
        pts1_s = feats0['keypoints'][m[:, 0]].cpu().numpy()   # 模板（匹配尺度坐标）
        pts2_s = feats1['keypoints'][m[:, 1]].cpu().numpy()
    log(f"匹配点: {len(m)}")

    # ---------- 刚性初值：MAGSAC H + 空间一致性剔除 ----------
    def fit_h(p, q):
        Hm, _ = cv2.findHomography(p.astype(np.float32), q.astype(np.float32),
                                   cv2.USAC_MAGSAC, 3.0)
        return Hm

    P, U = pts1_s.astype(np.float64), (pts2_s - pts1_s).astype(np.float64)
    H_w = fit_h(P, U)
    from scipy.spatial import cKDTree
    for it in range(3):
        pred_p = cv2.perspectiveTransform(P.reshape(-1, 1, 2).astype(np.float32), H_w).reshape(-1, 2)
        res = (P + U) - pred_p
        tree = cKDTree(P)
        k = min(12, len(P))
        _, nb = tree.query(P, k=k)
        med = np.median(U[nb], axis=1)
        dev = np.linalg.norm(U - med, axis=1)
        mad = np.median(dev) + 1e-6
        good = (dev < max(6.0, 4 * mad)) & (np.linalg.norm(res, axis=1) < max(10.0, 6 * mad))
        log(f"  迭代{it+1}: 保留 {good.sum()}/{len(P)} (MAD={mad:.2f})")
        if good.sum() < 20 or good.all():
            break
        P, U = P[good], U[good]
        H_w = fit_h(P, U)
    H_w, mask = cv2.findHomography(P.astype(np.float32), (P + U).astype(np.float32),
                                   cv2.USAC_MAGSAC, 3.0)
    inl = mask.ravel().astype(bool)
    log(f"刚性内点: {inl.sum()}/{len(P)} ({inl.mean()*100:.1f}%)")
    P, U = P[inl], U[inl]

    # 升全分辨率，计算 H 后非刚性残差（TPS 拟合目标）
    pts1 = P / s
    S = np.diag([1/s, 1/s, 1]).astype(np.float64)
    H_full = S @ H_w.astype(np.float64) @ np.diag([s, s, 1])
    u = (P + U) / s - cv2.perspectiveTransform(pts1.reshape(-1, 1, 2), H_full).reshape(-1, 2)
    err_rigid = np.linalg.norm(u, axis=1)
    log(f"刚性 H 拟合残差(全分辨率px): median={np.median(err_rigid):.2f}, "
        f"P95={np.percentile(err_rigid, 95):.2f}")

    # ---------- 留出法选 λ ----------
    norm = np.array([FW, FH], dtype=np.float64)
    X = pts1 / norm
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    n_hold = max(30, int(len(X) * HOLDOUT_FRAC))
    te, tr = idx[:n_hold], idx[n_hold:]
    log("留出法扫描 λ ...")
    results = []
    for lam in LAMBDA_SWEEP:
        rbf = RBFInterpolator(X[tr], u[tr], kernel='thin_plate_spline',
                              smoothing=lam, neighbors=TPS_NEIGHBORS)
        err = np.linalg.norm(u[te] - rbf(X[te]), axis=1)
        rmse = float(np.sqrt((err ** 2).mean()))
        results.append((lam, rmse))
        log(f"  λ={lam:<8g} -> holdout RMSE={rmse:.2f}px")
    rmse_rigid_te = float(np.sqrt((err_rigid[te] ** 2).mean()))
    log(f"  (基线) 纯刚性 holdout RMSE={rmse_rigid_te:.2f}px")
    best_lam, best_rmse = min(results, key=lambda t: t[1])
    log(f"选定 λ={best_lam:g}")
    tps = RBFInterpolator(X, u, kernel='thin_plate_spline',
                          smoothing=best_lam, neighbors=TPS_NEIGHBORS)
    err_tps = np.linalg.norm(u - tps(X), axis=1)
    log(f"TPS 后残差: median={np.median(err_tps):.2f}, P95={np.percentile(err_tps, 95):.2f}px")

    # ---------- 稠密 remap + 可视化（与 NCC 版一致） ----------
    vw, vh = VIS_WIDTH, int(round(FH * VIS_WIDTH / FW))
    vs = FW / vw
    log(f"生成可视化尺度 {vw}x{vh} 的 remap 并 warp ...")
    tpl_vis = cv2.resize(tpl_bgr, (vw, vh), interpolation=cv2.INTER_AREA)
    smp_vis = cv2.resize(smp_bgr, (vw, vh), interpolation=cv2.INTER_AREA)
    ys, xs = np.mgrid[0:vh, 0:vw].astype(np.float64)
    q = np.stack([xs, ys], axis=-1).reshape(-1, 2) * vs
    Hi_full = np.linalg.inv(H_full)
    x0 = cv2.perspectiveTransform(q.reshape(-1, 1, 2), Hi_full).reshape(-1, 2)
    u_dense = tps(x0 / norm).reshape(vh, vw, 2)
    x1 = cv2.perspectiveTransform((q - u_dense.reshape(-1, 2)).reshape(-1, 1, 2),
                                  Hi_full).reshape(-1, 2)
    map_tps = (x1 / vs).astype(np.float32).reshape(vh, vw, 2)
    warped_tps = cv2.remap(tpl_vis, map_tps, None, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    x1r = cv2.perspectiveTransform(q.reshape(-1, 1, 2), Hi_full).reshape(-1, 2)
    map_rigid = (x1r / vs).astype(np.float32).reshape(vh, vw, 2)
    warped_rigid = cv2.remap(tpl_vis, map_rigid, None, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    g_smp = cv2.cvtColor(smp_vis, cv2.COLOR_BGR2GRAY)
    g_rigid = cv2.cvtColor(warped_rigid, cv2.COLOR_BGR2GRAY)
    g_tps = cv2.cvtColor(warped_tps, cv2.COLOR_BGR2GRAY)

    def stack_overlay(a, b):
        out = np.zeros((a.shape[0], a.shape[1], 3), np.uint8)
        out[..., 0] = b; out[..., 1] = a
        return out

    def diff_map(a, b, gain=6):
        return cv2.convertScaleAbs(cv2.absdiff(a, b), alpha=gain)

    cv2.imwrite(os.path.join(OUT_DIR, "1_overlay_rigid.png"), stack_overlay(g_smp, g_rigid))
    cv2.imwrite(os.path.join(OUT_DIR, "2_overlay_tps.png"), stack_overlay(g_smp, g_tps))
    cv2.imwrite(os.path.join(OUT_DIR, "3_diff_rigid.png"), diff_map(g_smp, g_rigid))
    cv2.imwrite(os.path.join(OUT_DIR, "4_diff_tps.png"), diff_map(g_smp, g_tps))

    e_rigid = float(((g_smp.astype(np.float64) - g_rigid.astype(np.float64)) ** 2).mean())
    e_tps = float(((g_smp.astype(np.float64) - g_tps.astype(np.float64)) ** 2).mean())

    # 位移场可视化
    mag, ang = cv2.cartToPolar(u_dense[..., 0].astype(np.float32),
                               u_dense[..., 1].astype(np.float32))
    hsv = np.zeros((vh, vw, 3), np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    m95 = float(np.percentile(mag, 95)) + 1e-6
    hsv[..., 2] = np.clip(mag / m95 * 255, 0, 255).astype(np.uint8)
    field_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    edge = cv2.Canny(g_smp, 60, 140)
    field_bgr[edge > 0] = (255, 255, 255)
    cv2.imwrite(os.path.join(OUT_DIR, "5_displacement_field.png"), field_bgr)

    # 全分辨率放大对比
    log("挑选改善最大的区域生成放大对比 ...")
    gs = 64
    gh, gw = vh // gs + 1, vw // gs + 1
    gain_grid = np.full((gh, gw), -1.0)
    d_r = cv2.absdiff(g_smp, g_rigid).astype(np.float64)
    d_t = cv2.absdiff(g_smp, g_tps).astype(np.float64)
    for gy in range(gh):
        for gx in range(gw):
            sl = np.s_[gy*gs:(gy+1)*gs, gx*gs:(gx+1)*gs]
            gain_grid[gy, gx] = d_r[sl].mean() - d_t[sl].mean()
    k2 = int(np.ceil(CROP_SIZE * vs / gs))
    gain_smooth = uniform_filter(gain_grid, size=k2)
    b = k2 // 2
    gain_smooth[:b+1] = -1; gain_smooth[-b-1:] = -1
    gain_smooth[:, :b+1] = -1; gain_smooth[:, -b-1:] = -1
    crops = []
    g = gain_smooth.copy()
    for i in range(N_CROPS):
        gy, gx = np.unravel_index(np.argmax(g), g.shape)
        g[max(0, gy-k2):gy+k2+1, max(0, gx-k2):gx+k2+1] = -1
        cx, cy = int(gx*gs*vs), int(gy*gs*vs)
        cx = min(max(cx, 0), FW - CROP_SIZE); cy = min(max(cy, 0), FH - CROP_SIZE)
        crops.append((cx, cy))
    sc = 1.0 / vs
    rows = []
    for (cx, cy) in crops:
        w_c, h_c = int(CROP_SIZE*sc), int(CROP_SIZE*sc)
        vx, vy = int(cx*sc), int(cy*sc)
        def tag(img, text, color):
            img = img.copy()
            cv2.rectangle(img, (0, 0), (img.shape[1]-1, img.shape[0]-1), color, 3)
            cv2.putText(img, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            return img
        row = np.hstack([tag(smp_vis[vy:vy+h_c, vx:vx+w_c], "sample", (255, 255, 255)),
                         tag(warped_rigid[vy:vy+h_c, vx:vx+w_c], "rigid H", (0, 0, 255)),
                         tag(warped_tps[vy:vy+h_c, vx:vx+w_c], "H + TPS", (0, 255, 0))])
        rows.append(row)
    sep = np.full((6, rows[0].shape[1], 3), 40, np.uint8)
    comp = rows[0]
    for r in rows[1:]:
        comp = np.vstack([comp, sep, r])
    cv2.imwrite(os.path.join(OUT_DIR, "6_fullres_crops_compare.png"), comp)

    summary = (
        f"\n===== 汇总（ALIKED+LightGlue 前端） =====\n"
        f"匹配点: {len(pts1_s)} -> 过滤后 {len(P)}\n"
        f"刚性 H 拟合残差(全分辨率px): median={np.median(err_rigid):.2f} P95={np.percentile(err_rigid,95):.2f}\n"
        f"TPS 后残差:                 median={np.median(err_tps):.2f} P95={np.percentile(err_tps,95):.2f}\n"
        f"留出 RMSE: 刚性={rmse_rigid_te:.2f}px  TPS(λ={best_lam:g})={best_rmse:.2f}px\n"
        f"差异能量 MSE(可视化尺度): 刚性={e_rigid:.1f} -> TPS={e_tps:.1f} "
        f"(降幅 {100*(1-e_tps/max(e_rigid,1e-9)):.1f}%)\n"
        f"位移场幅度(全分辨率px): median={float(np.median(mag))*vs:.1f} P95={m95*vs:.1f}\n"
        f"输出目录: {OUT_DIR}\n"
    )
    print(summary)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary)
    log("完成")


if __name__ == "__main__":
    main()

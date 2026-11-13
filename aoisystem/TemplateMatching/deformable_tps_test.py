# -*- coding: utf-8 -*-
"""
可变形匹配离线验证：刚性 H 初值 + 稠密 NCC 位移场 + TPS 正则
对照实验：单全局 H（现状） vs H + TPS 位移场（提案）
输入：object_region_full_res.jpg（模板） / corresponding_sample_region_full_res.jpg（采样图）
输出：对齐叠加图、差异图、位移场可视化、全分辨率局部放大对比、精度指标
"""
import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree
import os, sys, time

TEMPLATE_PATH = r"D:\Users\Administrator\Desktop\weights222\object_region_full_res.jpg"
SAMPLE_PATH   = r"D:\Users\Administrator\Desktop\weights222\corresponding_sample_region_full_res.jpg"
OUT_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deformable_tps_test_output")

# ---------- 参数 ----------
WORK_FACTOR   = 2          # 在半分辨率上做 NCC 匹配（精度损失=2px全分辨率，速度×4）
COARSE_TILE   = 300        # 粗对齐大 tile（半分辨率 px）
COARSE_STEP   = 400
COARSE_SCORE  = 0.55
DENSE_TILE    = 160        # 稠密位移场小 tile
DENSE_STEP    = 96
DENSE_SEARCH  = 60         # 局部搜索半径（围绕刚性预测位置）
DENSE_SCORE   = 0.50
TPS_NEIGHBORS = 64         # 局部化 TPS：限制位移场最短波长（抗缺陷吸收）
LAMBDA_SWEEP  = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
HOLDOUT_FRAC  = 0.25
VIS_WIDTH     = 3000
CROP_SIZE     = 700        # 全分辨率放大窗口
N_CROPS       = 4


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ncc_match(tpl_patch, search_img):
    """在 search_img 中找 tpl_patch 最佳位置，返回 (x, y, score) 亚像素坐标（patch 中心）"""
    if tpl_patch.shape[0] >= search_img.shape[0] or tpl_patch.shape[1] >= search_img.shape[1]:
        return None
    res = cv2.matchTemplate(search_img, tpl_patch, cv2.TM_CCOEFF_NORMED)
    _, _, _, peak = cv2.minMaxLoc(res)
    px, py = peak
    # 亚像素抛物线拟合
    def subpix(c, axis):
        if axis == 0:
            if 0 < c < res.shape[1]-1:
                d = res[int(np.clip(py,0,res.shape[0]-1)), c-1] - 2*res[int(np.clip(py,0,res.shape[0]-1)), c] + res[int(np.clip(py,0,res.shape[0]-1)), c+1]
                o = res[int(np.clip(py,0,res.shape[0]-1)), c-1] - res[int(np.clip(py,0,res.shape[0]-1)), c+1]
            else: return 0.0
        else:
            if 0 < c < res.shape[0]-1:
                d = res[c-1, int(np.clip(px,0,res.shape[1]-1))] - 2*res[c, int(np.clip(px,0,res.shape[1]-1))] + res[c+1, int(np.clip(px,0,res.shape[1]-1))]
                o = res[c-1, int(np.clip(px,0,res.shape[1]-1))] - res[c+1, int(np.clip(px,0,res.shape[1]-1))]
            else: return 0.0
        return 0.0 if d <= 0 else float(np.clip(0.5*o/d, -1, 1))
    dx = subpix(px, 0); dy = subpix(py, 1)
    th, tw = tpl_patch.shape
    cx = px + dx + tw/2.0
    cy = py + dy + th/2.0
    return cx, cy, float(res[py, px])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log("加载全分辨率图像 ...")
    tpl_bgr = cv2.imread(TEMPLATE_PATH)
    smp_bgr = cv2.imread(SAMPLE_PATH)
    FH, FW = tpl_bgr.shape[:2]
    log(f"模板/采样图尺寸: {FW}x{FH}")

    wh, ww = FH // WORK_FACTOR, FW // WORK_FACTOR
    tpl_w = cv2.cvtColor(cv2.resize(tpl_bgr, (ww, wh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    smp_w = cv2.cvtColor(cv2.resize(smp_bgr, (ww, wh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    log(f"工作分辨率: {ww}x{wh}")

    # ---------- 1. 稠密局部 NCC：在恒等位置附近搜索（相位相关确认两图基本同框） ----------
    log("稠密位移场估计：小 tile 局部 NCC（恒等位置附近搜索）...")
    t2 = DENSE_TILE // 2
    d_pts1, d_u, d_score = [], [], []
    for ty in range(t2, wh - t2, DENSE_STEP):
        for tx in range(t2, ww - t2, DENSE_STEP):
            x0, y0 = tx - DENSE_SEARCH - t2, ty - DENSE_SEARCH - t2
            x1, y1 = tx + DENSE_SEARCH + t2, ty + DENSE_SEARCH + t2
            if x0 < 0 or y0 < 0 or x1 > ww or y1 > wh:
                continue
            patch = tpl_w[ty-t2:ty+t2, tx-t2:tx+t2]
            m = ncc_match(patch, smp_w[y0:y1, x0:x1])
            if m and m[2] > DENSE_SCORE:
                d_pts1.append((tx, ty))
                d_u.append((m[0] + x0 - tx, m[1] + y0 - ty))   # 残差位移（半分辨率 px）
                d_score.append(m[2])
    d_pts1 = np.float64(d_pts1); d_u = np.float64(d_u); d_score = np.float64(d_score)
    log(f"稠密点: {len(d_pts1)}, NCC分数 median={np.median(d_score):.2f}")

    # ---------- 2. 刚性初值：分数过滤 -> RANSAC 单应 -> 空间一致性剔除 -> 重拟合 ----------
    keep = d_score > 0.60
    P, U = d_pts1[keep], d_u[keep]
    log(f"分数>0.60 的点: {len(P)}/{len(d_pts1)}")

    def fit_h(p, u_):
        Hm, _ = cv2.findHomography(p.astype(np.float32), (p + u_).astype(np.float32),
                                   cv2.USAC_MAGSAC, 3.0)
        return Hm

    H_w = fit_h(P, U)
    for it in range(3):
        pred = cv2.perspectiveTransform(P.reshape(-1, 1, 2).astype(np.float32), H_w).reshape(-1, 2)
        res = (P + U) - pred
        tree = cKDTree(P)
        k = min(12, len(P))
        _, nb = tree.query(P, k=k)
        med = np.median(U[nb], axis=1)
        dev = np.linalg.norm(U - med, axis=1)
        mad = np.median(dev) + 1e-6
        good = (dev < max(6.0, 4 * mad)) & (np.linalg.norm(res, axis=1) < max(10.0, 6 * mad))
        log(f"  迭代{it+1}: 空间一致性剔除后保留 {good.sum()}/{len(P)} (MAD={mad:.2f})")
        if good.sum() < 20 or good.all():
            break
        P, U = P[good], U[good]
        H_w = fit_h(P, U)
    H_w, mask = cv2.findHomography(P.astype(np.float32), (P + U).astype(np.float32),
                                   cv2.USAC_MAGSAC, 3.0)
    inl_c = mask.ravel().astype(bool)
    log(f"刚性初值内点: {inl_c.sum()}/{len(P)} ({inl_c.mean()*100:.1f}%)")
    if inl_c.sum() < 8:
        log("粗对齐失败，中止"); sys.exit(1)
    d_pts1 = P[inl_c]; d_u = U[inl_c]
    d_score = np.full(len(d_pts1), 1.0)
    # 升到全分辨率坐标
    pts1 = d_pts1 * WORK_FACTOR
    u_total = d_u * WORK_FACTOR          # 恒等位置的原始位移（含刚性分量，仅用于场可视化参照）
    # H 之后的非刚性残差（TPS 的真正拟合目标）
    pred_w = cv2.perspectiveTransform(d_pts1.reshape(-1, 1, 2).astype(np.float32),
                                      H_w).reshape(-1, 2)
    u = ((d_pts1 + d_u) - pred_w) * WORK_FACTOR
    S = np.diag([WORK_FACTOR, WORK_FACTOR, 1]).astype(np.float64)
    H_full = S @ H_w.astype(np.float64) @ np.diag([1/WORK_FACTOR, 1/WORK_FACTOR, 1])
    err_rigid = np.linalg.norm(u, axis=1)
    log(f"刚性 H 拟合残差(全分辨率px): median={np.median(err_rigid):.2f}, "
        f"P95={np.percentile(err_rigid, 95):.2f}, max={err_rigid.max():.2f}")

    # ---------- 3. 留出法自动选 λ ----------
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
    log(f"  (基线) λ=∞(纯刚性) holdout RMSE={rmse_rigid_te:.2f}px")

    best_lam, best_rmse = min(results, key=lambda t: t[1])
    log(f"选定 λ={best_lam:g}")

    tps = RBFInterpolator(X, u, kernel='thin_plate_spline',
                          smoothing=best_lam, neighbors=TPS_NEIGHBORS)
    err_tps = np.linalg.norm(u - tps(X), axis=1)
    log(f"TPS 后残差(全点): median={np.median(err_tps):.2f}, P95={np.percentile(err_tps, 95):.2f}px")

    # ---------- 4. 稠密 remap：模板 -> 采样帧（可视化尺度） ----------
    vw, vh = VIS_WIDTH, int(round(FH * VIS_WIDTH / FW))
    vs = FW / vw
    log(f"生成可视化尺度 {vw}x{vh} 的 remap 并 warp ...")
    tpl_vis = cv2.resize(tpl_bgr, (vw, vh), interpolation=cv2.INTER_AREA)
    smp_vis = cv2.resize(smp_bgr, (vw, vh), interpolation=cv2.INTER_AREA)

    ys, xs = np.mgrid[0:vh, 0:vw].astype(np.float64)
    q = np.stack([xs, ys], axis=-1).reshape(-1, 2) * vs        # 全分辨率采样坐标
    Hi_full = np.linalg.inv(H_full)
    x0 = cv2.perspectiveTransform(q.reshape(-1, 1, 2), Hi_full).reshape(-1, 2)
    u_dense = tps(x0 / norm).reshape(vh, vw, 2)                # 全分辨率 px
    # remap 目标：采样像素 q 的来源 = H^-1(q - u)
    x1 = cv2.perspectiveTransform((q - u_dense.reshape(-1, 2)).reshape(-1, 1, 2), Hi_full).reshape(-1, 2)
    map_tps = (x1 / vs).astype(np.float32).reshape(vh, vw, 2)
    warped_tps = cv2.remap(tpl_vis, map_tps, None, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # 纯刚性对照：u=0
    x1r = cv2.perspectiveTransform(q.reshape(-1, 1, 2), Hi_full).reshape(-1, 2)
    map_rigid = (x1r / vs).astype(np.float32).reshape(vh, vw, 2)
    warped_rigid = cv2.remap(tpl_vis, map_rigid, None, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def stack_overlay(a_gray, b_gray):
        h, w = a_gray.shape
        out = np.zeros((h, w, 3), np.uint8)
        out[..., 0] = b_gray   # B: 模板(warped)
        out[..., 1] = a_gray   # G: 采样
        return out

    def diff_map(a_gray, b_gray, gain=6):
        d = cv2.absdiff(a_gray, b_gray)
        return cv2.convertScaleAbs(d, alpha=gain)

    g_smp = cv2.cvtColor(smp_vis, cv2.COLOR_BGR2GRAY)
    g_rigid = cv2.cvtColor(warped_rigid, cv2.COLOR_BGR2GRAY)
    g_tps = cv2.cvtColor(warped_tps, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(os.path.join(OUT_DIR, "1_overlay_rigid.png"), stack_overlay(g_smp, g_rigid))
    cv2.imwrite(os.path.join(OUT_DIR, "2_overlay_tps.png"), stack_overlay(g_smp, g_tps))
    cv2.imwrite(os.path.join(OUT_DIR, "3_diff_rigid.png"), diff_map(g_smp, g_rigid))
    cv2.imwrite(os.path.join(OUT_DIR, "4_diff_tps.png"), diff_map(g_smp, g_tps))

    e_rigid = float(((g_smp.astype(np.float64) - g_rigid.astype(np.float64)) ** 2).mean())
    e_tps = float(((g_smp.astype(np.float64) - g_tps.astype(np.float64)) ** 2).mean())
    log(f"差异能量 MSE(可视化尺度): 刚性={e_rigid:.1f}, TPS={e_tps:.1f}, 降幅={100*(1-e_tps/max(e_rigid,1e-9)):.1f}%")

    # ---------- 5. 位移场可视化 ----------
    mag, ang = cv2.cartToPolar(u_dense[..., 0].astype(np.float32), u_dense[..., 1].astype(np.float32))
    hsv = np.zeros((vh, vw, 3), np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    m95 = float(np.percentile(mag, 95)) + 1e-6
    hsv[..., 2] = np.clip(mag / m95 * 255, 0, 255).astype(np.uint8)
    field_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    edge = cv2.Canny(g_smp, 60, 140)
    field_bgr[edge > 0] = (255, 255, 255)
    txt = f"|u| P95={m95*vs:.1f}px full-res (hue=dir, bright=large)"
    cv2.putText(field_bgr, txt, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 6)
    cv2.putText(field_bgr, txt, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(OUT_DIR, "5_displacement_field.png"), field_bgr)

    # ---------- 6. 全分辨率局部放大对比 ----------
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
    k = int(np.ceil(CROP_SIZE * vs / gs))
    gain_smooth = uniform_filter(gain_grid, size=k)
    b = k // 2
    gain_smooth[:b+1] = -1; gain_smooth[-b-1:] = -1
    gain_smooth[:, :b+1] = -1; gain_smooth[:, -b-1:] = -1

    crops = []
    g = gain_smooth.copy()
    for i in range(N_CROPS):
        gy, gx = np.unravel_index(np.argmax(g), g.shape)
        g[max(0, gy-k):gy+k+1, max(0, gx-k):gx+k+1] = -1
        cx, cy = int(gx*gs*vs), int(gy*gs*vs)
        cx = min(max(cx, 0), FW - CROP_SIZE); cy = min(max(cy, 0), FH - CROP_SIZE)
        crops.append((cx, cy))

    sc = 1.0 / vs
    rows = []
    for (cx, cy) in crops:
        w_c, h_c = int(CROP_SIZE*sc), int(CROP_SIZE*sc)
        vx, vy = int(cx*sc), int(cy*sc)
        smp_c = smp_vis[vy:vy+h_c, vx:vx+w_c]
        rig_c = warped_rigid[vy:vy+h_c, vx:vx+w_c]
        tps_c = warped_tps[vy:vy+h_c, vx:vx+w_c]
        def tag(img, text, color):
            img = img.copy()
            cv2.rectangle(img, (0, 0), (img.shape[1]-1, img.shape[0]-1), color, 3)
            cv2.putText(img, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            return img
        row = np.hstack([tag(smp_c, "sample", (255, 255, 255)),
                         tag(rig_c, "rigid H", (0, 0, 255)),
                         tag(tps_c, "H + TPS", (0, 255, 0))])
        rows.append(row)
    sep = np.full((6, rows[0].shape[1], 3), 40, np.uint8)
    comp = rows[0]
    for r in rows[1:]:
        comp = np.vstack([comp, sep, r])
    cv2.imwrite(os.path.join(OUT_DIR, "6_fullres_crops_compare.png"), comp)

    # ---------- 汇总 ----------
    summary = (
        f"\n===== 汇总 =====\n"
        f"稠密位移点(过滤后): {len(d_pts1)}\n"
        f"刚性 H 残差(全分辨率px): median={np.median(err_rigid):.2f} P95={np.percentile(err_rigid,95):.2f}\n"
        f"TPS 后残差:              median={np.median(err_tps):.2f} P95={np.percentile(err_tps,95):.2f}\n"
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

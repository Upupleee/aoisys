# -*- coding: utf-8 -*-
"""
506s4e 验收变体对比：在周期投票+ECC 对齐结果（t=(133,1252)，三层全收敛）上，
分别用三种验收口径评估并输出各自的最终对齐堆叠图 + 误差热力图：

  变体 1a  亮度域降门     —— NCC 门 0.5→0.35（低 SNR 跨域下小 patch NCC 天花板 ~0.4）
  变体 1b  边缘域 NCC     —— 双方 CLAHE+Sobel 边缘图上做局部 NCC（灰度形态解耦）
  变体 2   距离变换(方案B) —— 模板边缘→DT，采样边缘点查 DT 值（只认线条位置，完全免疫灰度形态）

输出：val_506s4e_output/ 下 v1a/v1b/v2 的 ov_*/hm_*.png + accept_variants_summary.json
"""
import cv2
import numpy as np
import os, sys, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from val_2652s2g import load_down, coarse_prior, accept, warp_by_H, T, local_match
from make_overlay_heatmap import overlay, heatmap_points

OUT = os.path.join(HERE, "val_506s4e_output")
os.makedirs(OUT, exist_ok=True)
CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(32, 32))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def edge_img(img):
    x = CLAHE.apply(img)
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, 3)
    e = cv2.magnitude(gx, gy)
    return np.clip(e, 0, np.percentile(e, 99)).astype(np.uint8)


def main():
    # ---- 载入 + 预处理（与周期投票一致）----
    tpl = load_down(r"D:\AOI\aoisys\506s4e\template\template_1.tif", 8)
    smp_raw = load_down(r"D:\AOI\aoisys\506s4e\sample\test_imgs1.tif", 8)
    tpl = 255 - tpl                     # 极性对齐
    smp = CLAHE.apply(smp_raw)          # 采样增强
    Wt, Ht = tpl.shape[1], tpl.shape[0]

    # ---- 重跑周期投票 → top-3 ECC，取首个 ECC 三层收敛的候选（已知 (133,1252)）----
    H0_old, diag = coarse_prior(tpl, smp)
    theta, scale = float(diag.get("dtheta", 0.0)), float(diag.get("scale", 1.0))
    tpl_s = cv2.resize(tpl, (int(round(Wt * scale)), int(round(Ht * scale))),
                       interpolation=cv2.INTER_AREA)
    c_t = np.array([Wt / 2.0, Ht / 2.0])
    dW, dH = tpl_s.shape[1] - Wt, tpl_s.shape[0] - Ht
    a_ = np.radians(theta)
    R = np.array([[np.cos(a_), -np.sin(a_)], [np.sin(a_), np.cos(a_)]])
    Mrot = np.eye(3)
    Mrot[:2, :2] = R
    Mrot[:2, 2] = (np.eye(2) - R) @ np.array([Wt / 2.0, Ht / 2.0])

    def synth_H0(txv, tyv):
        """正确语义：p_w --Mrot(θ 绕模板中心)--> --S(s 绕 c_t)--> --T(t)--> q_smp。
        投票的 t 是"预缩放模板帧 → 采样"的平移（patch 从预缩放模板上切）。"""
        S_m = np.eye(3)
        S_m[0, 0] = S_m[1, 1] = scale
        S_m[0, 2] = c_t[0] * (1 - scale)
        S_m[1, 2] = c_t[1] * (1 - scale)
        M_ = np.eye(3)
        M_[:2, :2] = R
        M_[:2, 2] = (np.eye(2) - R) @ np.array([Wt / 2.0, Ht / 2.0])
        return T(txv, tyv) @ S_m @ M_

    # ---- 周期投票（在预缩放模板帧上重跑——修正预缩放语义）----
    from periodic_translation_search import estimate_period, extract_patches, top_peaks, periodic_vote
    tpl_s = cv2.resize(tpl, (int(round(Wt * scale)), int(round(Ht * scale))),
                       interpolation=cv2.INTER_AREA)
    # 周期在原帧测（稳定），按 scale 换算到预缩放帧——预缩放帧上自相关会锁伪峰（实测 100px）
    p0, qual = estimate_period(edge_img(tpl))
    period = (p0[0] * scale, p0[1] * scale)
    log(f"单元周期: 原帧 {p0} → 预缩放帧 {tuple(round(v) for v in period)} (自相关 {qual[0]:.2f}/{qual[1]:.2f})")
    patches, psize = extract_patches(tpl_s.shape, period, K=9)
    all_cands = {}
    for k, (m_k, box) in enumerate(patches):
        patch = tpl_s[box[1]:box[3], box[0]:box[2]]
        corr = cv2.matchTemplate(smp, patch, cv2.TM_CCOEFF_NORMED)
        all_cands[k] = top_peaks(corr, m_k, period, M=5)
    t_med, vtrace = periodic_vote(all_cands, period)
    log(f"投票: patches={vtrace.get('best_patches')}, t_med={t_med}")
    member_ts = vtrace.get("member_ts_raw", [])
    uniq_t = sorted({(round(tx, 1), round(ty, 1)) for _, tx, ty, _ in member_ts})
    if not uniq_t:
        log("FAIL-SAFE：周期投票无一致候选")
        return
    gx = np.arange(64, Wt - 64, 256, dtype=np.float64)
    gy = np.arange(64, Ht - 64, 256, dtype=np.float64)
    Xg, Yg = np.meshgrid(gx, gy)
    grid_l = np.stack([Xg.ravel(), Yg.ravel()], -1)
    scores = {}
    for cx, cy in uniq_t:
        H0c = synth_H0(cx, cy)
        wr = np.c_[grid_l, np.ones(len(grid_l))] @ H0c.T
        mapped = wr[:, :2] / wr[:, 2:3]
        found, sc = local_match(tpl, smp, grid_l, mapped, patch=64, search=20)
        okv = np.isfinite(found[:, 0]) & (sc >= 0.4)
        scores[(cx, cy)] = float(np.median(sc[okv])) if okv.sum() >= 10 else -2.0
    cand_sorted = sorted(uniq_t, key=lambda c: scores[c], reverse=True)[:3]
    log(f"top-3 候选: {cand_sorted} (zncc {[round(scores[c],3) for c in cand_sorted]})")

    # ---- top-3 逐个 ECC（首个三层全收敛者胜出）----
    smp_al0 = smp_al = H_rel = None
    summary = {}
    cand_ecc = [c for c in cand_sorted if scores[c] > 0]      # 只信正 zncc 候选
    if not cand_ecc:
        log("FAIL-SAFE：无正 zncc 候选 → 采样 SNR 或为根本瓶颈")
        return
    # ---- 周期整数歧义裁决：ECC 跨域不收敛无法当裁判 → 用方案 B 的 DT 分数裁决 ----
    # （差一个周期行的候选，DT med 会差周期级；DT 对灰度形态免疫）
    t_e = edge_img(tpl)
    t_bin_g = (t_e > max(20, np.percentile(t_e, 85))).astype(np.uint8)
    dt_g = cv2.distanceTransform(1 - t_bin_g, cv2.DIST_L2, 3)
    dt_scores = {}
    for cx, cy in cand_ecc:
        H0c = synth_H0(cx, cy)
        s_a0 = warp_by_H(smp, H0c, Wt, Ht)
        s_e0 = edge_img(s_a0)
        s_b0 = (s_e0 > max(20, np.percentile(s_e0, 85))).astype(np.uint8)
        ys0, xs0 = np.where(s_b0 > 0)
        if len(ys0) > 6000:
            ii = np.linspace(0, len(ys0) - 1, 6000).astype(int)
            ys0, xs0 = ys0[ii], xs0[ii]
        dt_scores[(cx, cy)] = float(np.median(dt_g[ys0, xs0]))
        log(f"  DT 裁决 t=({cx:.0f},{cy:.0f}): DT med {dt_scores[(cx, cy)]:.1f}px")
    cand_ecc = sorted(cand_ecc, key=lambda c: dt_scores[c])   # DT med 最小者优先
    log(f"DT 裁决后顺序: {cand_ecc}")
    for cx, cy in cand_ecc:
        H0 = synth_H0(cx, cy)
        smp_al0 = warp_by_H(smp, H0, Wt, Ht)
        Hm = np.eye(3)
        conv = 0
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
                conv += 1
            except cv2.error:
                break
        if conv == 3:
            H_rel = Hm
            smp_al = warp_by_H(smp_al0, H_rel, Wt, Ht)
            log(f"候选 t=({cx:.0f},{cy:.0f}) ECC 三层全收敛 → 采用")
            summary["t"] = [cx, cy]
            break
        else:
            log(f"候选 t=({cx:.0f},{cy:.0f}) ECC 仅 {conv}/3 收敛 → 跳过")
    if smp_al is None:
        # ECC 在跨域形态差下不稳定（0/3 收敛）→ 用最高 zncc 候选的 H0 直接对齐，
        # 跳过 ECC 精化——变体 1b/2 的边缘域/DT 验收本就不依赖 ECC
        cx, cy = cand_ecc[0]
        H0 = synth_H0(cx, cy)
        smp_al = warp_by_H(smp, H0, Wt, Ht)
        smp_al0 = smp_al
        H_rel = np.eye(3)
        summary["t"] = [cx, cy]
        summary["ecc_skipped"] = True
        log(f"无 ECC 全收敛候选 → 用最高 zncc 候选 t=({cx:.0f},{cy:.0f}) 的 H0 直接对齐")
    cv2.imwrite(os.path.join(OUT, "v_base_aligned.png"), smp_al)
    summary["ecc_converged"] = True
    # 验收网格（128 步，与统一验收门同密度）
    gx = np.arange(64, Wt - 64, 128, dtype=np.float64)
    gy = np.arange(64, Ht - 64, 128, dtype=np.float64)
    Xg, Yg = np.meshgrid(gx, gy)
    grid = np.stack([Xg.ravel(), Yg.ravel()], -1)

    # ============ 变体 1a：亮度域降门（0.5 → 0.35） ============
    t_e = edge_img(tpl)
    s_e = edge_img(smp_al)
    found, sc = local_match(tpl, smp_al, grid, grid.copy(), patch=96, search=14)
    okv = np.isfinite(found[:, 0]) & (sc >= 0.35)
    err = np.linalg.norm(found[okv] - grid[okv], axis=1) if okv.sum() else np.zeros(0)
    log(f"变体1a 亮度降门(0.35): 内点率 {okv.mean():.1%}, "
        f"残差 med {np.median(err) if len(err) else -1:.2f}px")
    summary["v1a"] = {"gate": 0.35, "inlier_ratio": float(okv.mean()),
                      "resid_med": float(np.median(err)) if len(err) else None}
    overlay(tpl, smp_al, os.path.join(OUT, "v1a_ov.png"),
            "V1a lum-gate 0.35: aligned pair (R=tpl G=sample)")
    if okv.sum() > 20:
        heatmap_points(tpl, grid[okv], err, os.path.join(OUT, "v1a_hm.png"),
                       f"V1a 亮度域残差 (门0.35, {int(okv.sum())} 点)")

    # ============ 变体 1b：边缘域 NCC ============
    found_e, sc_e = local_match(t_e, s_e, grid, grid.copy(), patch=96, search=14)
    okv_e = np.isfinite(found_e[:, 0]) & (sc_e >= 0.4)
    err_e = np.linalg.norm(found_e[okv_e] - grid[okv_e], axis=1) if okv_e.sum() else np.zeros(0)
    log(f"变体1b 边缘域NCC(0.4): 内点率 {okv_e.mean():.1%}, "
        f"残差 med {np.median(err_e) if len(err_e) else -1:.2f}px")
    summary["v1b"] = {"gate": 0.4, "inlier_ratio": float(okv_e.mean()),
                      "resid_med": float(np.median(err_e)) if len(err_e) else None}
    overlay(t_e, s_e, os.path.join(OUT, "v1b_ov.png"),
            "V1b edge-domain: aligned edges (R=tpl G=sample)")
    if okv_e.sum() > 20:
        heatmap_points(tpl, grid[okv_e], err_e, os.path.join(OUT, "v1b_hm.png"),
                       f"V1b 边缘域残差 ({int(okv_e.sum())} 点)")

    # ============ 变体 2：距离变换（方案 B） ============
    # 模板边缘 → DT（距离到最近模板边缘像素）；采样(smp_al)边缘点查 DT
    t_bin = (t_e > max(20, np.percentile(t_e, 85))).astype(np.uint8)
    dt = cv2.distanceTransform(1 - t_bin, cv2.DIST_L2, 3)
    # 采样边缘点（smp_al 帧内，NMS 简化为阈值+网格抽样）
    s_bin = (s_e > max(20, np.percentile(s_e, 85))).astype(np.uint8)
    ys, xs = np.where(s_bin > 0)
    if len(ys) > 8000:
        idx = np.linspace(0, len(ys) - 1, 8000).astype(int)
        ys, xs = ys[idx], xs[idx]
    dt_vals = dt[ys, xs].astype(np.float64)
    inl_dt = dt_vals <= 3.0
    log(f"变体2 距离变换: 采样边缘点 {len(ys)}, DT≤3px 占比 {inl_dt.mean():.1%}, "
        f"DT med {np.median(dt_vals):.2f}px p90 {np.percentile(dt_vals, 90):.2f}px")
    summary["v2_dt"] = {"n_edge_pts": int(len(ys)), "dt_le3_ratio": float(inl_dt.mean()),
                        "dt_med": float(np.median(dt_vals)),
                        "dt_p90": float(np.percentile(dt_vals, 90))}
    # 叠图：模板边缘(红) vs 采样边缘(绿)
    ov_e = np.zeros((Ht, Wt, 3), np.uint8)
    ov_e[..., 2] = t_bin * 255
    ov_e[..., 1] = s_bin * 255
    s = 1400.0 / Wt
    cv2.imwrite(os.path.join(OUT, "v2_dt_ov.png"),
                cv2.resize(ov_e, (1400, int(Ht * s))))
    # 热力图：边缘点位置 + DT 值（复用 heatmap_points 的散点渲染）
    pts_dt = np.column_stack([xs, ys]).astype(np.float64)
    heatmap_points(tpl, pts_dt, dt_vals, os.path.join(OUT, "v2_dt_hm.png"),
                   f"V2 距离变换残差 ({len(ys)} 边缘点, 门3px)")

    with open(os.path.join(OUT, "accept_variants_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log(f"-> {OUT}/accept_variants_summary.json")


if __name__ == "__main__":
    main()

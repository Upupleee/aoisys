# -*- coding: utf-8 -*-
"""
方案 A′：周期一致平移搜索（506s4e 低 SNR 采样 + 重复单元板的全局定位）。

问题：模板内容在采样中偏移大（~(296,1439)px 级），跨域下全幅相位相关峰弱（resp≈0.009）；
     单 patch 匹配在重复单元阵列上多峰（各锁不同单元，平移差 = 单元周期整数倍）。

解法（报告 §11.10 方案 A′）：
  1. 模板取反（极性对齐）+ 采样 CLAHE；
  2. 模板自相关/FFT 投影自动估计单元周期 (px,py)；
  3. 取 K 个分散位置的单元 patch，各在采样上 matchTemplate 取 top-M 局部峰
     （峰间距 ≥ 周期/2，抑制同单元邻域峰）；
  4. 周期投票：所有候选平移 t_c = 峰 − patch锚点，对周期取模入桶（桶宽=周期/10）；
     真平移对所有 patch 一致 → 同桶聚集；单 patch 误锁被多数票否决；
  5. 最高桶 patch 覆盖率 ≥ 60% 通过；桶内以最高分峰为锚、其余 ±n·周期 对齐后取中位数；
  6. t + FM(θ,s) 合成 H0 → 矫正 → ECC 金字塔 → NCC 三遍 → 统一验收门。

用法：python periodic_translation_search.py [pair_no=1] [dataset_base=506s4e]
"""
import cv2
import numpy as np
import os, sys, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from val_2652s2g import load_down, coarse_prior, accept, warp_by_H, T, local_match
from retest_phase_corr import compose_H
from classical_feature_test import refine, mag_h
import synthetic_tps_validation as stv
from make_overlay_heatmap import overlay, heatmap_points

OUT = os.path.join(HERE, "val_506s4e_output")
os.makedirs(OUT, exist_ok=True)
CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(32, 32))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------- 1. 单元周期自动估计（模板投影自相关） ----------------

def estimate_period(tpl_edge):
    """对模板边缘图做水平/垂直投影的自相关，在合理周期区间 [100, n/2] 内找主峰。
    （不限范围会锁到梳齿内部细结构的伪周期，实测 16px）"""
    def first_period(sig):
        sig = sig - sig.mean()
        n = len(sig)
        f = np.fft.rfft(sig)
        ac = np.fft.irfft(f * np.conj(f))[: n // 2]
        ac /= ac[0] + 1e-9
        lo, hi = 100, max(200, n // 2)
        seg = ac[lo:hi]
        k = int(np.argmax(seg)) + lo
        return k, float(ac[k])

    px_, qx = first_period(tpl_edge.sum(axis=0))   # 水平周期
    py_, qy = first_period(tpl_edge.sum(axis=1))   # 垂直周期
    return (px_, py_), (qx, qy)


# ---------------- 2. 分散 patch + top-M 局部峰 ----------------

def extract_patches(tpl_shape, period, K=9, cover=1.6):
    """3x3 分散网格取 patch；patch 尺寸 = 周期的 cover 倍（覆盖一个完整单元）。
    返回 [(锚点m_k, patch)]，锚点 = patch 左上角（模板系）。"""
    h, w = tpl_shape
    pw = int(min(w * 0.35, period[0] * cover))
    ph = int(min(h * 0.25, period[1] * cover))
    pw, ph = max(pw, 128), max(ph, 128)
    patches = []
    for gy in (0.12, 0.5, 0.88):
        for gx in (0.12, 0.5, 0.88):
            mx = int(gx * (w - pw))
            my = int(gy * (h - ph))
            patches.append(((mx, my), (mx, my, mx + pw, my + ph)))
    return patches[:K], (pw, ph)


def top_peaks(corr, m_k, period, M=5, min_dist=None):
    """相关面上取 top-M 局部极大（峰间距 ≥ max(周期/2, 64px)）。"""
    if min_dist is None:
        min_dist = max(int(min(period) / 2), 64)
    r = corr.copy()
    peaks = []
    for _ in range(M * 3):
        _, mv, _, loc = cv2.minMaxLoc(r)
        if mv <= 0:
            break
        x, y = loc
        if all((x - px) ** 2 + (y - py) ** 2 >= min_dist ** 2 for px, py, _ in peaks):
            peaks.append((x, y, float(mv)))
            if len(peaks) >= M:
                break
        # 抑制邻域
        x0, y0 = max(x - min_dist, 0), max(y - min_dist, 0)
        r[y0:y + min_dist, x0:x + min_dist] = -1
    # 候选平移 t_c = 峰位置 − patch 锚点
    return [(x - m_k[0], y - m_k[1], s) for x, y, s in peaks]


# ---------------- 3. 周期投票 ----------------

def periodic_vote(all_cands, period, bucket_frac=0.1, need_ratio=0.6):
    """所有候选平移对周期取模入桶（mod 周期，防止桶边界分裂一致平移）；
    最高桶的 patch 覆盖率 ≥ need_ratio 通过。
    all_cands: {patch_idx: [(t_cx, t_cy, score)]}"""
    bx = max(int(period[0] * bucket_frac), 8)
    by = max(int(period[1] * bucket_frac), 8)
    buckets = {}
    for k, cands in all_cands.items():
        for tx, ty, s in cands:
            key = (int(round((tx % period[0]) / bx)), int(round((ty % period[1]) / by)))
            buckets.setdefault(key, []).append((k, tx, ty, s))
    # 统计每个桶覆盖的 patch 数
    best_key, best_patches = None, -1
    for key, members in buckets.items():
        nk = len({m[0] for m in members})
        if nk > best_patches:
            best_patches, best_key = nk, key
    members = buckets.get(best_key, [])
    patch_ids = {m[0] for m in members}
    ok = best_patches >= max(2, int(np.ceil(len(all_cands) * need_ratio)))
    trace = {"buckets_top": sorted(
        [{"key": k, "patches": len({m[0] for m in ms}), "votes": len(ms)}
         for k, ms in buckets.items()],
        key=lambda d: -d["patches"])[:5]}
    if not ok:
        return None, trace
    # 记录桶内成员的"原始绝对平移"——mod 一致但整数周期 n 可能不同，
    # 锚峰锁错周期行会全组整体错位 → 交由外部逐成员轻量打分裁决（周期整数歧义）
    trace["member_ts_raw"] = [(k, tx, ty, s) for k, tx, ty, s in members]
    anchor = max(members, key=lambda m: m[3])
    aligned = []
    for k, tx, ty, s in members:
        nx = round((tx - anchor[1]) / period[0])
        ny = round((ty - anchor[2]) / period[1])
        aligned.append((tx - nx * period[0], ty - ny * period[1]))
    t = (float(np.median([a[0] for a in aligned])),
         float(np.median([a[1] for a in aligned])))
    return t, {**trace, "best_patches": best_patches, "members": len(members),
               "anchor_score": anchor[3], "t": t, "ok": True}


# ---------------- 主链路 ----------------

def main(pair=1, base=r"D:\AOI\aoisys\506s4e"):
    ti = pair
    tag = f"p{ti}"
    tpl_raw = load_down(os.path.join(base, "template", f"template_{ti}.tif"), 8)
    smp_raw = load_down(os.path.join(base, "sample", f"test_imgs{ti}.tif"), 8)
    tpl = 255 - tpl_raw                    # 极性对齐（黑底白线）
    smp = CLAHE.apply(smp_raw)             # 采样增强
    Wt, Ht = tpl.shape[1], tpl.shape[0]
    log(f"工作分辨率 tpl {Wt}x{Ht}, smp {smp.shape[1]}x{smp.shape[0]}")

    # 1. FM (θ,s)——只用角度缩放，不用其平移
    H0_old, diag = coarse_prior(tpl, smp)
    theta, scale = float(diag.get("dtheta", 0.0)), float(diag.get("scale", 1.0))
    log(f"FM: θ={theta}° s={scale}")

    # 2. 周期估计
    t_e = prep_edge(tpl)
    s_e = prep_edge(smp)
    period, qual = estimate_period(t_e)
    log(f"单元周期估计: {period} (自相关 {qual[0]:.2f}/{qual[1]:.2f})")

    # 3. 多 patch + top-M 峰（亮度域：取反模板 patch vs CLAHE 采样——实测峰 0.33~0.58 的配置）
    patches, psize = extract_patches(tpl.shape, period, K=9)
    all_cands = {}
    for k, (m_k, box) in enumerate(patches):
        patch = tpl[box[1]:box[3], box[0]:box[2]]          # 取反亮度图切 patch
        corr = cv2.matchTemplate(smp, patch, cv2.TM_CCOEFF_NORMED)   # CLAHE 采样亮度图
        all_cands[k] = top_peaks(corr, m_k, period, M=5)
    log(f"候选峰: {['%d' % len(v) for v in all_cands.values()]}")

    # 4. 周期投票
    t, vtrace = periodic_vote(all_cands, period)
    log(f"投票: {vtrace}")
    if t is None:
        log("投票未通过（无一致平移）→ FAIL-SAFE")
        return

    # 5. 合成 H0（封装，供周期整数歧义裁决复用）：缩放吸收进模板预放大，H0 = 平移 t + 小角度旋转
    tpl_s = cv2.resize(tpl, (int(round(Wt * scale)), int(round(Ht * scale))),
                       interpolation=cv2.INTER_AREA)
    c_t = np.array([Wt / 2.0, Ht / 2.0])
    dW, dH = tpl_s.shape[1] - Wt, tpl_s.shape[0] - Ht
    a_ = np.radians(theta)
    R = np.array([[np.cos(a_), -np.sin(a_)], [np.sin(a_), np.cos(a_)]])
    Mrot = np.eye(3)
    Mrot[:2, :2] = R
    Mrot[:2, 2] = (np.eye(2) - R) @ np.array([Wt / 2.0, Ht / 2.0])
    Sinv = np.linalg.inv(np.array([[scale, 0, c_t[0] * (1 - scale)],
                                   [0, scale, c_t[1] * (1 - scale)], [0, 0, 1]], dtype=np.float64))

    def synth_H0(txv, tyv):
        return (T(txv, tyv) @ Mrot @ T(-dW / 2.0, -dH / 2.0)) @ Sinv

    # 5.5 周期整数歧义裁决：mod 一致但整数周期 n 可能不同。
    #     轻量打分只做相对排序（粗网格在低 SNR 上无法区分差一个周期的候选——
    #     实测 y 方向三候选 zncc 0.478/0.427/0.464 几乎相同），最终由 ECC+统一验收门裁决。
    member_ts = vtrace.get("member_ts_raw", [])
    uniq_t = sorted({(round(tx, 1), round(ty, 1)) for _, tx, ty, _ in member_ts})
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
        scores[(cx, cy)] = (float(np.median(sc[okv])) if okv.sum() >= 10 else -2.0)
        log(f"  候选 t=({cx:.0f},{cy:.0f}): 内点率 {okv.mean():.1%} zncc_med {scores[(cx, cy)]:.3f}")
    cand_sorted = sorted(uniq_t, key=lambda c: scores[c], reverse=True)[:3]
    log(f"周期整数裁决: top-3 候选 {cand_sorted} → 逐个走 ECC+验收门")

    # 6. top-3 候选逐个：矫正 → ECC → 验收门（首个通过者胜出）
    smp_al0 = H_rel = m1 = pts1 = err1 = None
    ok1 = False
    chosen = None
    for cx, cy in cand_sorted:
        H0 = synth_H0(cx, cy)
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
                log(f"  ECC f={f}: 收敛")
            except cv2.error:
                log(f"  ECC f={f}: 失败")
                break
        H_rel = Hm
        ok1, m1, pts1, err1 = accept(tpl, smp_al0, H_rel)
        log(f"  候选 t=({cx:.0f},{cy:.0f}) ECC后验收: {'通过' if ok1 else '未通过'} {m1}")
        if ok1:
            chosen = (cx, cy)
            break
    if not ok1:
        log("FAIL-SAFE：周期定位后仍未过验收门 → 采样 SNR 或为根本瓶颈")
        return
    log(f"胜出平移: t={chosen}")

    smp_al = warp_by_H(smp_al0, H_rel, Wt, Ht)
    Hm2 = np.eye(3)
    pts = disp = None
    for srch, patch_sz, ncc_min in [(stv.SEARCH_R, stv.PATCH, stv.NCC_MIN),
                                    (48, stv.PATCH, stv.NCC_MIN),
                                    (stv.SMALL_SEARCH, stv.SMALL_PATCH, stv.SMALL_NCC)]:
        p_, d_, sc_ = stv.match_grid(tpl, smp_al, H0=Hm2, search_r=srch, patch=patch_sz)
        keep = stv.consistency_filter(p_, d_, sc_, ncc_min=ncc_min)
        log(f"  NCC pass: 内点 {int(keep.sum())}/{len(p_)}")
        if keep.sum() < 30:
            pts = None
            break
        pts, disp = p_[keep], d_[keep]
        Hm2, _ = stv.fit_homography(pts, disp)
    if pts is None:
        log("NCC 三遍失败")
        return
    pred = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32),
                                    Hm2).reshape(-1, 2)
    err = np.linalg.norm(pts + disp - pred, axis=1)
    log(f"NCC 三遍: {len(pts)} 点, med {np.median(err):.2f}px p95 {np.percentile(err, 95):.2f}px")
    ok2, m2, *_ = accept(tpl, smp_al, Hm2)
    log(f"NCC-H 验收: {'通过' if ok2 else '未通过'} {m2}")
    overlay(tpl, warp_by_H(smp_al, Hm2, Wt, Ht),
            os.path.join(OUT, f"{tag}_ov_periodic.png"),
            "506 p1: periodic vote + ECC + NCC (aligned)")
    heatmap_points(tpl, pts, err, os.path.join(OUT, f"{tag}_hm_periodic.png"),
                   f"周期投票链路 NCC 残差 ({len(pts)} 点)")
    with open(os.path.join(OUT, f"{tag}_periodic_result.json"), "w", encoding="utf-8") as f:
        json.dump({"period": period, "period_quality": qual, "vote": vtrace,
                   "t": t, "fm": diag, "ecc": m1, "ncc": m2,
                   "final_ok": bool(ok2)}, f, indent=2, ensure_ascii=False)
    log(f"-> {OUT}/{tag}_periodic_result.json")


def prep_edge(img):
    x = CLAHE.apply(img)
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, 3)
    e = cv2.magnitude(gx, gy)
    return np.clip(e, 0, np.percentile(e, 99)).astype(np.uint8)


if __name__ == "__main__":
    pair = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    base = sys.argv[2] if len(sys.argv) > 2 else r"D:\AOI\aoisys\506s4e"
    main(pair, base)

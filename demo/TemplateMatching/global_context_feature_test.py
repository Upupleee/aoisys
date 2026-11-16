# -*- coding: utf-8 -*-
"""
给 SIFT/ORB 注入全局信息的对照实验：空间门控迭代重匹配。
Stage0: 标准 ratio test 基线 -> MAGSAC H0
Stage1..2: 每个模板关键点在 H0 预测位置 ±WIN 内筛选候选，再做窗内 ratio test -> MAGSAC 更新 H
对比：匹配数、内点率、精化残差、最终 H 与 NCC 基准的稠密偏差。
"""
import cv2
import numpy as np
import os, time, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import synthetic_tps_validation as stv
from classical_feature_test import refine, mag_h, resid_stats

OUT_DIR = os.path.join(HERE, "classical_feature_output")
os.makedirs(OUT_DIR, exist_ok=True)
DOWNSCALE = 0.5
MAX_KPTS = 8000
RATIO_GLOBAL = 0.75
WIN = 24.0            # 空间门控窗口(px)
RATIO_LOCAL = 0.9     # 窗内 ratio 阈值放宽（真身已唯一，歧义被几何消掉）


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def spatial_gated(pt_kp, ps_cands, ds_cands, idx_cands, H0, win=WIN, ratio=RATIO_LOCAL):
    """对每个模板点，取 H0 预测位置 ±win 内的候选，窗内 ratio test。
    返回 (模板点索引数组, 训练图关键点 trainIdx 数组)。"""
    sel_t, sel_s = [], []
    for i in range(len(pt_kp)):
        p = pt_kp[i]
        w = H0 @ np.array([p[0], p[1], 1.0])
        px, py = w[0] / w[2], w[1] / w[2]
        cand = [(d.distance, j, pt) for d, j, pt in zip(ds_cands[i], idx_cands[i], ps_cands[i])
                if abs(pt[0] - px) <= win and abs(pt[1] - py) <= win]
        if not cand:
            continue
        cand.sort(key=lambda c: c[0])
        if len(cand) >= 2:
            if cand[0][0] < ratio * cand[1][0]:
                sel_t.append(i)
                sel_s.append(cand[0][1])
        else:
            sel_t.append(i)
            sel_s.append(cand[0][1])
    return np.array(sel_t, int), np.array(sel_s, int)


def main():
    log("读取图像…")
    tpl0 = cv2.imread(stv.TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
    smp0 = cv2.imread(r"D:\Users\Administrator\Desktop\weights222\corresponding_sample_region_full_res.jpg",
                      cv2.IMREAD_GRAYSCALE)
    Wf, Hf = int(tpl0.shape[1] * DOWNSCALE), int(tpl0.shape[0] * DOWNSCALE)
    tpl = cv2.resize(tpl0, (Wf, Hf), interpolation=cv2.INTER_AREA)
    smp = cv2.resize(smp0, (Wf, Hf), interpolation=cv2.INTER_AREA)

    # NCC 基准 H（评价参照）
    pts, disp, sc = stv.match_grid(tpl, smp)
    keep = stv.consistency_filter(pts, disp, sc)
    H_ncc, _ = stv.fit_homography(pts[keep], disp[keep])

    def dense_dev(Ha):
        xs = np.arange(0, Wf, 16, dtype=np.float64)
        ys = np.arange(0, Hf, 16, dtype=np.float64)
        X, Y = np.meshgrid(xs, ys)
        t = np.stack([X.ravel(), Y.ravel()], axis=-1)
        a = np.c_[t, np.ones(len(t))] @ Ha.T
        b = np.c_[t, np.ones(len(t))] @ H_ncc.T
        d = np.linalg.norm(a[:, :2] / a[:, 2:3] - b[:, :2] / b[:, 2:3], axis=1)
        return float(np.median(d)), float(np.sqrt((d ** 2).mean()))

    detectors = {
        "SIFT": cv2.SIFT_create(nfeatures=MAX_KPTS),
        "ORB": cv2.ORB_create(nfeatures=MAX_KPTS, fastThreshold=10, patchSize=64),
    }
    results = {}
    for name, det in detectors.items():
        log(f"== {name} ==")
        kt, dt_ = det.detectAndCompute(tpl, None)
        ks, ds = det.detectAndCompute(smp, None)
        norm = cv2.NORM_HAMMING if name == "ORB" else cv2.NORM_L2
        bf = cv2.BFMatcher(norm)
        pairs = bf.knnMatch(dt_, ds, k=10)
        # 基线：全局 ratio(0.75, k=2 语义从 k=10 的前二取)
        base = []
        for p in pairs:
            if len(p) >= 2 and p[0].distance < RATIO_GLOBAL * p[1].distance:
                base.append(p[0])
        pt_b = np.array([kt[m.queryIdx].pt for m in base], np.float64)
        ps_b = np.array([ks[m.trainIdx].pt for m in base], np.float64)
        # 初值用相似变换（4 DOF，少点数下远比单应稳健）
        A0, inl_a = cv2.estimateAffinePartial2D(pt_b.reshape(-1, 1, 2).astype(np.float32),
                                                ps_b.reshape(-1, 1, 2).astype(np.float32),
                                                method=cv2.RANSAC, ransacReprojThreshold=6.0,
                                                maxIters=10000)
        inl_b = inl_a.ravel().astype(bool)
        H0 = np.eye(3)
        H0[:2, :2], H0[:2, 2] = A0[:, :2], A0[:, 2]
        log(f"基线: 匹配 {len(base)}, 内点 {int(inl_b.sum())} ({inl_b.mean():.1%})")

        # 预取每个模板点的 top-10 候选（训练图坐标 / DMatch / trainIdx）
        ps_cands = [[ks[m.trainIdx].pt for m in p] for p in pairs]
        ds_cands = [[m for m in p] for p in pairs]
        idx_cands = [[m.trainIdx for m in p] for p in pairs]
        pt_all = np.array([kt[p[0].queryIdx].pt for p in pairs], np.float64) if pairs else np.zeros((0, 2))

        t0 = time.time()
        Hg = H0
        hist = []
        for it in range(2):
            ti, si = spatial_gated(pt_all, ps_cands, ds_cands, idx_cands, Hg)
            if len(ti) < 20:
                log(f"  迭代{it+1}: 门控匹配不足 ({len(ti)})")
                break
            ptg = pt_all[ti]
            psg = np.array([ks[j].pt for j in si], np.float64)
            Hnew, inl_g = mag_h(ptg, psg)
            if Hnew is None:
                log(f"  迭代{it+1}: MAGSAC 退化，保持上一轮 H")
                inl_g = np.ones(len(ti), bool)
                Hnew = Hg
            Hg = Hnew
            dev_m, dev_r = dense_dev(Hg)
            hist.append(dict(it=it + 1, n=len(ti), inliers=int(inl_g.sum()),
                             ratio=float(inl_g.mean()), dev_med=dev_m, dev_rms=dev_r))
            log(f"  迭代{it+1}: 门控匹配 {len(ti)}, 内点 {int(inl_g.sum())} ({inl_g.mean():.1%}), "
                f"H vs NCC dev med={dev_m:.2f}px RMS={dev_r:.2f}px")
        t_gate = time.time() - t0

        # 精化 + 残差
        oki = inl_g
        pr, nr = refine(tpl, smp, ptg[oki], psg[oki])
        okn = nr >= 0.5
        Hf2, inl_f = mag_h(ptg[oki][okn], pr[okn])
        rs = resid_stats(ptg[oki][okn][inl_f], pr[okn][inl_f], Hf2)
        dev_m, dev_r = dense_dev(Hf2)
        log(f"精化后: 内点 {int(inl_f.sum())}, resid med/RMS {rs['median']:.2f}/{rs['rms']:.2f}px, "
            f"H dev med={dev_m:.2f}px; 门控耗时 {t_gate:.2f}s")
        results[name] = dict(baseline=dict(matches=len(base), inliers=int(inl_b.sum())),
                             gated=hist, refined=dict(inliers=int(inl_f.sum()), resid=rs,
                                                      dev_med=dev_m, dev_rms=dev_r, t=round(t_gate, 2)))

    out = json.dumps(results, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, "global_context_results.json"), "w", encoding="utf-8") as f:
        f.write(out)
    print(out)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
纯 DL（ALIKED+LightGlue，不加 NCC 精化）的定位误差实测：
1) 每个匹配点：DL 原始坐标 vs NCC 亚像素精化坐标（高置信参照）的偏差分布
2) 全局：纯 DL 单应 vs 精化单应 在稠密网格上的像素差
"""
import cv2
import numpy as np
import os, time, json

TEMPLATE_PATH = r"D:\Users\Administrator\Desktop\weights222\object_region_full_res.jpg"
SAMPLE_PATH = r"D:\Users\Administrator\Desktop\weights222\corresponding_sample_region_full_res.jpg"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "dl_match_output")
DOWNSCALE = 0.5
MAX_KPTS = 8000
REFINE_PATCH, REFINE_SEARCH, NCC_MIN = 64, 16, 0.5


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def subpixel_peak(score):
    py, px = np.unravel_index(np.argmax(score), score.shape)
    if py <= 0 or px <= 0 or py >= score.shape[0] - 1 or px >= score.shape[1] - 1:
        return float(px), float(py), float(score[py, px])
    def para(m1, c, p1):
        den = (m1 - 2 * c + p1)
        return 0.0 if den == 0 else 0.5 * (m1 - p1) / den
    return px + para(score[py, px - 1], score[py, px], score[py, px + 1]), \
        py + para(score[py - 1, px], score[py, px], score[py + 1, px]), float(score[py, px])


def refine_one(tpl, smp, pts_t, pts_s):
    h, w = tpl.shape
    pad = REFINE_SEARCH + 2
    smp_pad = cv2.copyMakeBorder(smp, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    half = REFINE_PATCH // 2
    hp = REFINE_PATCH + 2 * pad
    out = np.zeros_like(pts_s)
    score = np.zeros(len(pts_t), np.float64)
    for i, ((tx, ty), (sx, sy)) in enumerate(zip(pts_t, pts_s)):
        tx, ty, sx, sy = int(round(tx)), int(round(ty)), int(round(sx)), int(round(sy))
        x0t, y0t = tx - half, ty - half
        x0s, y0s = sx - half - REFINE_SEARCH, sy - half - REFINE_SEARCH
        if x0t < 0 or y0t < 0 or x0t + REFINE_PATCH > w or y0t + REFINE_PATCH > h:
            continue
        if x0s < 0 or y0s < 0 or x0s + hp > smp_pad.shape[1] or y0s + hp > smp_pad.shape[0]:
            continue
        patch = tpl[y0t:y0t + REFINE_PATCH, x0t:x0t + REFINE_PATCH]
        win = smp_pad[y0s:y0s + hp, x0s:x0s + hp]
        sc = cv2.matchTemplate(win, patch, cv2.TM_CCOEFF_NORMED)
        mx, my, pk = subpixel_peak(sc)
        out[i] = (x0s + mx - pad + half, y0s + my - pad + half)
        score[i] = pk
    return out, score


def main():
    log("读取图像…")
    tpl0 = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
    smp0 = cv2.imread(SAMPLE_PATH, cv2.IMREAD_GRAYSCALE)
    Wf, Hf = int(tpl0.shape[1] * DOWNSCALE), int(tpl0.shape[0] * DOWNSCALE)
    tpl = cv2.resize(tpl0, (Wf, Hf), interpolation=cv2.INTER_AREA)
    smp = cv2.resize(smp0, (Wf, Hf), interpolation=cv2.INTER_AREA)

    import torch
    from lightglue import LightGlue
    from lightglue.aliked import ALIKED
    extractor = ALIKED(max_num_keypoints=MAX_KPTS).eval()
    matcher = LightGlue(features="aliked").eval()

    def to_tensor(img):
        return torch.from_numpy(img.astype(np.float32) / 255.0)[None, None]

    with torch.inference_mode():
        f_t = extractor.extract(to_tensor(tpl))
        f_s = extractor.extract(to_tensor(smp))
        pred = matcher({"image0": f_t, "image1": f_s})
        m = pred["matches"][0].cpu().numpy()
        pts_t = f_t["keypoints"][0].cpu().numpy().astype(np.float64)[m[:, 0]]
        pts_s = f_s["keypoints"][0].cpu().numpy().astype(np.float64)[m[:, 1]]
    log(f"DL 匹配 {len(m)} 对")

    pts_s_ref, ncc_sc = refine_one(tpl, smp, pts_t, pts_s)
    ok = ncc_sc >= 0.7   # 高置信参照
    err = np.linalg.norm(pts_s - pts_s_ref, axis=1)

    # 按参照置信度分层
    layers = {}
    for name, mask in [("ncc>=0.9", ok & (ncc_sc >= 0.9)), ("ncc0.7-0.9", ok & (ncc_sc < 0.9)), ("ncc<0.7(不可信)", ~ok & (ncc_sc >= 0))]:
        e = err[mask]
        if len(e):
            layers[name] = dict(n=int(len(e)),
                                median=float(np.median(e)), rms=float(np.sqrt(np.mean(e ** 2))),
                                p95=float(np.percentile(e, 95)), p99=float(np.percentile(e, 99)),
                                mx=float(e.max()))

    # 纯DL H vs 精化H 稠密像素差
    def fit(pt, ps):
        Hm, mask = cv2.findHomography(pt.reshape(-1, 1, 2).astype(np.float32),
                                      ps.reshape(-1, 1, 2).astype(np.float32),
                                      cv2.USAC_MAGSAC, 6.0, maxIters=10000)
        return Hm, mask.ravel().astype(bool)

    H_raw, inl_raw = fit(pts_t, pts_s)
    H_ref, _ = fit(pts_t[ok], pts_s_ref[ok])
    xs = np.arange(0, Wf, 8, dtype=np.float64)
    ys = np.arange(0, Hf, 8, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    t = np.stack([X.ravel(), Y.ravel()], axis=-1)
    a = np.c_[t, np.ones(len(t))] @ H_raw.T
    b = np.c_[t, np.ones(len(t))] @ H_ref.T
    d = np.linalg.norm(a[:, :2] / a[:, 2:3] - b[:, :2] / b[:, 2:3], axis=1)
    dense = dict(median=float(np.median(d)), rms=float(np.sqrt(np.mean(d ** 2))),
                 p95=float(np.percentile(d, 95)), mx=float(d.max()))

    res = dict(quant_step_px_at_half=float(7645 / 1024 / 2), layers=layers,
               H_diff_dense=dense,
               inlier_resid_raw=dict(n=int(inl_raw.sum()),
                                     median=float(np.median(np.linalg.norm(
                                         pts_s[inl_raw] - cv2.perspectiveTransform(
                                             pts_t[inl_raw].reshape(-1, 1, 2).astype(np.float32),
                                             H_raw).reshape(-1, 2), axis=1)))))
    with open(os.path.join(OUT_DIR, "dl_only_eval.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    lines = ["纯 DL (ALIKED+LightGlue, 无精化) 定位误差 @0.5x（全分辨率需 ×2）", ""]
    lines.append(f"检测原生分辨率 1024px -> 半分辨率坐标放大系数 ~{7645/1024:.1f}x, 理论量化步长 ~{7645/1024/2*2:.1f}px@0.5x")
    lines.append("")
    lines.append(f"{'参照置信':<14}{'n':>6}{'median':>9}{'RMS':>8}{'P95':>8}{'P99':>8}{'max':>9}")
    for k, v in layers.items():
        lines.append(f"{k:<14}{v['n']:>6}{v['median']:>9.2f}{v['rms']:>8.2f}{v['p95']:>8.2f}{v['p99']:>8.2f}{v['mx']:>9.2f}")
    lines.append("")
    lines.append(f"纯DL单应 vs 精化单应 稠密像素差: median={dense['median']:.2f} RMS={dense['rms']:.2f} "
                 f"P95={dense['p95']:.2f} max={dense['mx']:.2f} px@0.5x")
    out = "\n".join(lines)
    print(out)
    with open(os.path.join(OUT_DIR, "dl_only_summary.txt"), "w", encoding="utf-8") as f:
        f.write(out + "\n")

    # 误差分布直方图（高置信参照）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2))
    e_ok = err[ok]
    ax.hist(e_ok, bins=60, color="#4a7dbd", edgecolor="none")
    for v, c, lab in [(np.median(e_ok), "r", f"median {np.median(e_ok):.2f}"),
                      (np.percentile(e_ok, 95), "orange", f"P95 {np.percentile(e_ok,95):.2f}")]:
        ax.axvline(v, color=c, lw=1.2, label=lab)
    ax.set_xlabel("pure-DL localization error vs NCC-refined (px @0.5x)")
    ax.set_ylabel("matches")
    ax.set_title(f"ALIKED+LightGlue raw localization error (n={len(e_ok)})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "dl_only_error_hist.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
传统特征点配准对照：SIFT / ORB vs ALIKED+LightGlue（同口径）。
每个检测器：原始匹配 + MAGSAC -> NCC 亚像素精化 + MAGSAC -> 纯特征定位误差（以高置信NCC精化位为参照）。
"""
import cv2
import numpy as np
import os, time, json

TEMPLATE_PATH = r"D:\Users\Administrator\Desktop\weights222\object_region_full_res.jpg"
SAMPLE_PATH = r"D:\Users\Administrator\Desktop\weights222\corresponding_sample_region_full_res.jpg"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "classical_feature_output")
os.makedirs(OUT_DIR, exist_ok=True)
DOWNSCALE = 0.5
MAX_KPTS = 8000
RATIO = 0.75
REFINE_PATCH, REFINE_SEARCH = 64, 16
MAGSAC_THRES = 6.0


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


def refine(tpl, smp, pts_t, pts_s):
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


def mag_h(pt, ps):
    Hm, mask = cv2.findHomography(pt.reshape(-1, 1, 2).astype(np.float32),
                                  ps.reshape(-1, 1, 2).astype(np.float32),
                                  cv2.USAC_MAGSAC, MAGSAC_THRES, maxIters=10000)
    return Hm, mask.ravel().astype(bool)


def resid_stats(pt, ps, Hm):
    r = np.linalg.norm(ps - cv2.perspectiveTransform(
        pt.reshape(-1, 1, 2).astype(np.float32), Hm).reshape(-1, 2), axis=1)
    return dict(median=float(np.median(r)), rms=float(np.sqrt((r ** 2).mean())),
                p95=float(np.percentile(r, 95)))


def main():
    log("读取图像…")
    tpl0 = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
    smp0 = cv2.imread(SAMPLE_PATH, cv2.IMREAD_GRAYSCALE)
    Wf, Hf = int(tpl0.shape[1] * DOWNSCALE), int(tpl0.shape[0] * DOWNSCALE)
    tpl = cv2.resize(tpl0, (Wf, Hf), interpolation=cv2.INTER_AREA)
    smp = cv2.resize(smp0, (Wf, Hf), interpolation=cv2.INTER_AREA)

    detectors = {
        "SIFT": cv2.SIFT_create(nfeatures=MAX_KPTS),
        "ORB": cv2.ORB_create(nfeatures=MAX_KPTS, fastThreshold=10, patchSize=64),
    }

    results = {}
    for name, det in detectors.items():
        log(f"== {name} ==")
        t0 = time.time()
        kt, dt_ = det.detectAndCompute(tpl, None)
        ks, ds = det.detectAndCompute(smp, None)
        t_det = time.time() - t0
        if dt_ is None or ds is None or len(kt) < 10:
            log(f"{name}: 特征不足")
            continue
        norm = cv2.NORM_HAMMING if name == "ORB" else cv2.NORM_L2
        bf = cv2.BFMatcher(norm)
        t0 = time.time()
        pairs = bf.knnMatch(dt_, ds, k=2)
        good = [m for m, n2 in (p for p in pairs if len(p) == 2) if m.distance < RATIO * n2.distance]
        t_match = time.time() - t0
        if len(good) < 10:
            log(f"{name}: 好匹配不足 ({len(good)})")
            continue
        pt = np.array([kt[m.queryIdx].pt for m in good], np.float64)
        ps = np.array([ks[m.trainIdx].pt for m in good], np.float64)

        H_raw, inl = mag_h(pt, ps)
        rs_raw = resid_stats(pt[inl], ps[inl], H_raw)

        ps_ref, nsc = refine(tpl, smp, pt, ps)
        okr = nsc >= 0.5
        H_ref, inl_r = mag_h(pt[okr], ps_ref[okr])
        rs_ref = resid_stats(pt[okr][inl_r], ps_ref[okr][inl_r], H_ref)

        # 纯特征定位误差（高置信参照）
        okc = nsc >= 0.9
        err = np.linalg.norm(ps - ps_ref, axis=1)
        loc = dict(n=int(okc.sum()), median=float(np.median(err[okc])),
                   rms=float(np.sqrt((err[okc] ** 2).mean())),
                   p95=float(np.percentile(err[okc], 95))) if okc.sum() > 10 else None

        t_total = t_det + t_match + (time.time() - t0) * 0  # det+match
        results[name] = dict(
            kpts_t=int(len(kt)), kpts_s=int(len(ks)), matches=len(good),
            det_s=round(t_det, 2), match_s=round(t_match, 2),
            raw=dict(inliers=int(inl.sum()), inlier_ratio=float(inl.mean()), resid=rs_raw),
            refined=dict(ncc_keep=int(okr.sum()), inliers=int(inl_r.sum()),
                         inlier_ratio=float(inl_r.mean()), resid=rs_ref),
            loc_err=loc)
        log(f"{name}: kpts {len(kt)}/{len(ks)}, 匹配 {len(good)} (det {t_det:.1f}s + match {t_match:.2f}s)")
        log(f"  raw:    内点 {inl.sum()} ({inl.mean():.1%}), resid {rs_raw['median']:.2f}/{rs_raw['rms']:.2f}/{rs_raw['p95']:.2f}")
        log(f"  refined: 内点 {inl_r.sum()} ({inl_r.mean():.1%}), resid {rs_ref['median']:.2f}/{rs_ref['rms']:.2f}/{rs_ref['p95']:.2f}")
        if loc:
            log(f"  纯特征定位误差(n={loc['n']}): med {loc['median']:.2f} / RMS {loc['rms']:.2f} / P95 {loc['p95']:.2f}px")

        # 匹配连线图
        s = 1400.0 / max(Hf, Hf)
        both = np.full((Hf, Wf * 2 + 8), 255, np.uint8)
        both[:, :Wf] = tpl
        both[:, Wf + 8:] = smp
        ov = cv2.cvtColor(cv2.resize(both, None, fx=s, fy=s), cv2.COLOR_GRAY2BGR)
        idx = np.linspace(0, len(pt) - 1, min(400, len(pt))).astype(int)
        for i in idx:
            if not inl[i]:
                continue
            p1 = (int(pt[i, 0] * s), int(pt[i, 1] * s))
            p2 = (int((ps[i, 0] + Wf + 8) * s), int(ps[i, 1] * s))
            cv2.line(ov, p1, p2, (0, 170, 0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUT_DIR, f"{name.lower()}_matches.png"), ov)

    # 汇总表（含 ALIKED 参照）
    lines = ["传统特征点 vs ALIKED+LightGlue @0.5x（同口径：原始匹配 / NCC精化后）", ""]
    lines.append(f"{'方法':<20}{'匹配':>6}{'内点率':>8}{'raw_med':>9}{'ref内点率':>10}{'ref_med':>9}{'ref_RMS':>9}{'定位med':>9}{'耗时s':>8}")
    def row(nm, m, ir, rm, ir2, rm2, t):
        return f"{nm:<20}{m:>6}{ir:>8.1%}{rm['median']:>9.2f}{ir2:>10.1%}{rm2['median']:>9.2f}{rm2['rms']:>9.2f}"
    for nm, r in results.items():
        loc = r["loc_err"]
        loc_s = f"{loc['median']:.2f}" if loc else "n/a"
        lines.append(f"{nm:<20}{r['matches']:>6}{r['raw']['inlier_ratio']:>8.1%}{r['raw']['resid']['median']:>9.2f}"
                     f"{r['refined']['inlier_ratio']:>10.1%}{r['refined']['resid']['median']:>9.2f}{r['refined']['resid']['rms']:>9.2f}"
                     f"{loc_s:>9}"
                     f"{r['det_s'] + r['match_s']:>8.2f}")
    lines.append(f"{'ALIKED+LG(raw)':<20}{4599:>6}{0.801:>8.1%}{2.36:>9.2f}{'':>10}{'':>9}{'':>9}{'':>9}{8.0:>8.2f}")
    lines.append(f"{'ALIKED+LG(refined)':<20}{4050:>6}{'':>8}{'':>9}{0.944:>10.1%}{0.74:>9.2f}{1.23:>9.2f}{'2.14*':>9}{9.0:>8.2f}")
    out = "\n".join(lines)
    print("\n" + out)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(out + "\n")
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

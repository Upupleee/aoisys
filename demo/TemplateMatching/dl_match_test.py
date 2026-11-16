# -*- coding: utf-8 -*-
"""
DL 特征匹配验证：ALIKED + LightGlue 在真实样本对（weights222）上的匹配效果。
流程：半分辨率 DL 提取/匹配 -> 局部 NCC 亚像素精化 -> MAGSAC 单应 -> 统计+可视化。
对比基准：现链路 NCC+MAGSAC = 1.03px / 2.5s（全分辨率）。
"""
import cv2
import numpy as np
import os, time, json

TEMPLATE_PATH = r"D:\Users\Administrator\Desktop\weights222\object_region_full_res.jpg"
SAMPLE_PATH = r"D:\Users\Administrator\Desktop\weights222\corresponding_sample_region_full_res.jpg"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "dl_match_output")
os.makedirs(OUT_DIR, exist_ok=True)

DOWNSCALE = 0.5
MAX_KPTS = 8000
REFINE_PATCH = 64     # NCC 亚像素精化 patch
REFINE_SEARCH = 16    # 精化搜索半径
NCC_MIN = 0.5
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


def refine_matches(tpl, smp, pts_t, pts_s):
    """局部 NCC 亚像素精化，返回精化后 pts_s' 与 ncc 分数。"""
    h, w = tpl.shape
    pad = REFINE_SEARCH + 2
    smp_pad = cv2.copyMakeBorder(smp, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    half = REFINE_PATCH // 2
    out = np.zeros_like(pts_s)
    score = np.zeros(len(pts_t), np.float64)
    hp = REFINE_PATCH + 2 * pad
    for i, ((tx, ty), (sx, sy)) in enumerate(zip(pts_t, pts_s)):
        tx, ty, sx, sy = int(round(tx)), int(round(ty)), int(round(sx)), int(round(sy))
        x0t, y0t = tx - half, ty - half
        if x0t < 0 or y0t < 0 or x0t + REFINE_PATCH > w or y0t + REFINE_PATCH > h:
            score[i] = -1
            continue
        patch = tpl[y0t:y0t + REFINE_PATCH, x0t:x0t + REFINE_PATCH]
        x0s = sx - half - REFINE_SEARCH
        y0s = sy - half - REFINE_SEARCH
        if x0s < 0 or y0s < 0 or x0s + hp > smp_pad.shape[1] or y0s + hp > smp_pad.shape[0]:
            score[i] = -1
            continue
        win = smp_pad[y0s:y0s + hp, x0s:x0s + hp]
        sc = cv2.matchTemplate(win, patch, cv2.TM_CCOEFF_NORMED)
        mx, my, pk = subpixel_peak(sc)
        out[i] = (x0s + mx - pad + half, y0s + my - pad + half)
        score[i] = pk
    return out, score


def magenta_green(tpl, rec):
    """模板=绿, 校正图=品红；对齐处灰白。"""
    return np.dstack([rec, tpl, rec])


def main():
    log("读取图像…")
    tpl0 = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
    smp0 = cv2.imread(SAMPLE_PATH, cv2.IMREAD_GRAYSCALE)
    if tpl0 is None or smp0 is None:
        raise SystemExit("图像读取失败")
    Wf = int(tpl0.shape[1] * DOWNSCALE)
    Hf = int(tpl0.shape[0] * DOWNSCALE)
    tpl = cv2.resize(tpl0, (Wf, Hf), interpolation=cv2.INTER_AREA)
    Ws = int(smp0.shape[1] * DOWNSCALE)
    Hs = int(smp0.shape[0] * DOWNSCALE)
    smp = cv2.resize(smp0, (Ws, Hs), interpolation=cv2.INTER_AREA)
    log(f"工作分辨率 tpl {Wf}x{Hf}, smp {Ws}x{Hs}")

    import torch
    from lightglue import LightGlue
    from lightglue.aliked import ALIKED
    device = torch.device("cpu")
    log(f"torch {torch.__version__}, device=cpu")

    extractor = ALIKED(max_num_keypoints=MAX_KPTS).eval().to(device)
    matcher = LightGlue(features="aliked").eval().to(device)

    def to_tensor(img):
        return torch.from_numpy(img.astype(np.float32) / 255.0)[None, None]

    t0 = time.time()
    with torch.inference_mode():
        f_t = extractor.extract(to_tensor(tpl))
        f_s = extractor.extract(to_tensor(smp))
        t_ext = time.time() - t0
        log(f"关键点: tpl {f_t['keypoints'].shape[1]}, smp {f_s['keypoints'].shape[1]}, 提取耗时 {t_ext:.1f}s")
        t0 = time.time()
        pred = matcher({"image0": f_t, "image1": f_s})
        t_match = time.time() - t0
        m = pred["matches"][0].cpu().numpy()
        kpts_t = f_t["keypoints"][0].cpu().numpy().astype(np.float64)
        kpts_s = f_s["keypoints"][0].cpu().numpy().astype(np.float64)
        pts_t = kpts_t[m[:, 0]]
        pts_s = kpts_s[m[:, 1]]
        log(f"LightGlue 匹配 {len(m)} 对, 匹配耗时 {t_match:.2f}s")

    # ---- 原始 DL 匹配 + MAGSAC ----
    H_raw, mask_raw = cv2.findHomography(pts_t.reshape(-1, 1, 2).astype(np.float32),
                                         pts_s.reshape(-1, 1, 2).astype(np.float32),
                                         cv2.USAC_MAGSAC, MAGSAC_THRES, maxIters=10000)
    inl_raw = mask_raw.ravel().astype(bool)
    resid_raw = np.linalg.norm(pts_s[inl_raw] - cv2.perspectiveTransform(
        pts_t[inl_raw].reshape(-1, 1, 2).astype(np.float32), H_raw).reshape(-1, 2), axis=1)

    # ---- NCC 亚像素精化 ----
    t0 = time.time()
    pts_s_ref, ncc_sc = refine_matches(tpl, smp, pts_t, pts_s)
    t_ref = time.time() - t0
    ok = ncc_sc >= NCC_MIN
    log(f"NCC 精化: {int(ok.sum())}/{len(ok)} 保留, 耗时 {t_ref:.1f}s")

    H_ref, mask_ref = cv2.findHomography(pts_t[ok].reshape(-1, 1, 2).astype(np.float32),
                                         pts_s_ref[ok].reshape(-1, 1, 2).astype(np.float32),
                                         cv2.USAC_MAGSAC, MAGSAC_THRES, maxIters=10000)
    inl_ref = mask_ref.ravel().astype(bool)
    resid_ref = np.linalg.norm(pts_s_ref[ok][inl_ref] - cv2.perspectiveTransform(
        pts_t[ok][inl_ref].reshape(-1, 1, 2).astype(np.float32), H_ref).reshape(-1, 2), axis=1)

    def st(e):
        return dict(median=float(np.median(e)), rms=float(np.sqrt(np.mean(e ** 2))),
                    p95=float(np.percentile(e, 95)))

    stats = dict(
        device=str(device), torch=torch.__version__,
        kpts_t=int(f_t["keypoints"].shape[1]), kpts_s=int(f_s["keypoints"].shape[1]),
        lg_matches=int(len(m)),
        raw=dict(inliers=int(inl_raw.sum()), inlier_ratio=float(inl_raw.mean()),
                 resid=st(resid_raw), extract_s=round(t_ext, 1), match_s=round(t_match, 2)),
        refined=dict(ncc_keep=int(ok.sum()), inliers=int(inl_ref.sum()),
                     inlier_ratio=float(inl_ref.mean()), resid=st(resid_ref),
                     refine_s=round(t_ref, 1)),
    )
    log(f"原始DL: 内点 {inl_raw.sum()}/{len(m)}, 内点残差 RMS={st(resid_raw)['rms']:.2f}px")
    log(f"精化DL: 内点 {inl_ref.sum()}, 内点残差 RMS={st(resid_ref)['rms']:.2f}px (中位 {st(resid_ref)['median']:.2f}px)")

    # ---- 可视化 1: 匹配线（降采样子集） ----
    s = 1400.0 / max(Hf, Hs)
    both = np.full((max(Hf, Hs), Wf + Ws + 8), 255, np.uint8)
    both[:Hf, :Wf] = tpl
    both[:Hs, Wf + 8:] = smp
    ov = cv2.cvtColor(cv2.resize(both, None, fx=s, fy=s), cv2.COLOR_GRAY2BGR)
    idx = np.linspace(0, len(pts_t) - 1, min(500, len(pts_t))).astype(int)
    for i in idx:
        if not (inl_raw[i] if len(inl_raw) == len(pts_t) else True):
            continue
        p1 = (int(pts_t[i, 0] * s), int(pts_t[i, 1] * s))
        p2 = (int((pts_s[i, 0] + Wf + 8) * s), int(pts_s[i, 1] * s))
        cv2.line(ov, p1, p2, (0, 160 if i % 2 else 255, 0), 1, cv2.LINE_AA)
        cv2.circle(ov, p1, 2, (0, 0, 255), -1)
        cv2.circle(ov, p2, 2, (255, 0, 0), -1)
    cv2.imwrite(os.path.join(OUT_DIR, "aliked_lightglue_matches.png"), ov)

    # ---- 可视化 2: 校正对齐（sample 按 H 反投影回模板系） ----
    for tag, Hm in (("raw", H_raw), ("refined", H_ref)):
        xs = np.arange(0, Wf, 2, dtype=np.float64)
        ys = np.arange(0, Hf, 2, dtype=np.float64)
        X, Y = np.meshgrid(xs, ys)
        t = np.stack([X.ravel(), Y.ravel()], axis=-1)
        th = np.c_[t, np.ones(len(t))] @ Hm.T
        hm = th[:, :2] / th[:, 2:3]
        rect = cv2.remap(smp, hm[:, 0].reshape(len(ys), len(xs)).astype(np.float32),
                         hm[:, 1].reshape(len(ys), len(xs)).astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        tpl_r = tpl[::2, ::2]
        ovl = magenta_green(tpl_r, rect)
        cv2.imwrite(os.path.join(OUT_DIR, f"aliked_overlay_{tag}_full.png"),
                    cv2.resize(ovl, None, fx=0.5, fy=0.5))
        # 中心 zoom
        ch, cw = tpl_r.shape[0] // 2, tpl_r.shape[1] // 2
        half = 500
        z_ovl = ovl[ch - half:ch + half, cw - half:cw + half]
        cv2.imwrite(os.path.join(OUT_DIR, f"aliked_overlay_{tag}_zoom.png"), z_ovl)
        ad = cv2.normalize(cv2.absdiff(tpl_r, rect), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imwrite(os.path.join(OUT_DIR, f"aliked_absdiff_{tag}.png"),
                    cv2.applyColorMap(ad, cv2.COLORMAP_INFERNO))

    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    lines = [f"ALIKED+LightGlue @ {DOWNSCALE}x, kpts={MAX_KPTS}, cpu",
             f"tpl kpts {stats['kpts_t']}, smp kpts {stats['kpts_s']}, LG matches {stats['lg_matches']}",
             f"raw:   inliers {stats['raw']['inliers']} ({stats['raw']['inlier_ratio']:.1%}), "
             f"resid median/rms/p95 = {stats['raw']['resid']['median']:.2f}/{stats['raw']['resid']['rms']:.2f}/{stats['raw']['resid']['p95']:.2f} px@0.5x",
             f"refined: inliers {stats['refined']['inliers']} ({stats['refined']['inlier_ratio']:.1%}), "
             f"resid median/rms/p95 = {stats['refined']['resid']['median']:.2f}/{stats['refined']['resid']['rms']:.2f}/{stats['refined']['resid']['p95']:.2f} px@0.5x",
             f"times: extract {t_ext:.1f}s, LG match {t_match:.2f}s, NCC refine {t_ref:.1f}s"]
    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

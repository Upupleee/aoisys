# -*- coding: utf-8 -*-
"""SuperPoint + LightGlue 真实对评测（与 ALIKED+LG 同口径）。输出 gapfill_superpoint_output/"""
import cv2
import numpy as np
import os, time, json

TEMPLATE_PATH = r"D:\Users\Administrator\Desktop\weights222\object_region_full_res.jpg"
SAMPLE_PATH = r"D:\Users\Administrator\Desktop\weights222\corresponding_sample_region_full_res.jpg"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "gapfill_superpoint_output")
os.makedirs(OUT_DIR, exist_ok=True)
DOWNSCALE, MAX_KPTS = 0.5, 8000
REFINE_PATCH, REFINE_SEARCH, NCC_MIN, MAGSAC_THRES = 64, 16, 0.5, 6.0


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


def main():
    log("读取图像…")
    tpl0 = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
    smp0 = cv2.imread(SAMPLE_PATH, cv2.IMREAD_GRAYSCALE)
    Wf, Hf = int(tpl0.shape[1] * DOWNSCALE), int(tpl0.shape[0] * DOWNSCALE)
    tpl = cv2.resize(tpl0, (Wf, Hf), interpolation=cv2.INTER_AREA)
    smp = cv2.resize(smp0, (Wf, Hf), interpolation=cv2.INTER_AREA)

    import torch
    from lightglue import LightGlue
    from lightglue.superpoint import SuperPoint
    extractor = SuperPoint(max_num_keypoints=MAX_KPTS).eval()
    matcher = LightGlue(features="superpoint").eval()

    def to_tensor(img):
        return torch.from_numpy(img.astype(np.float32) / 255.0)[None, None]

    with torch.inference_mode():
        t0 = time.time()
        f_t = extractor.extract(to_tensor(tpl))
        f_s = extractor.extract(to_tensor(smp))
        t_ext = time.time() - t0
        log(f"关键点 tpl {f_t['keypoints'].shape[1]} / smp {f_s['keypoints'].shape[1]}, 提取 {t_ext:.1f}s")
        t0 = time.time()
        pred = matcher({"image0": f_t, "image1": f_s})
        t_match = time.time() - t0
        m = pred["matches"][0].cpu().numpy()
        pts_t = f_t["keypoints"][0].cpu().numpy().astype(np.float64)[m[:, 0]]
        pts_s = f_s["keypoints"][0].cpu().numpy().astype(np.float64)[m[:, 1]]
        log(f"SP+LG 匹配 {len(m)} 对, 匹配 {t_match:.2f}s")

    def mag(pt, ps):
        Hm, mask = cv2.findHomography(pt.reshape(-1, 1, 2).astype(np.float32),
                                      ps.reshape(-1, 1, 2).astype(np.float32),
                                      cv2.USAC_MAGSAC, MAGSAC_THRES, maxIters=10000)
        return Hm, mask.ravel().astype(bool)

    def rs(pt, ps, Hm):
        r = np.linalg.norm(ps - cv2.perspectiveTransform(
            pt.reshape(-1, 1, 2).astype(np.float32), Hm).reshape(-1, 2), axis=1)
        return dict(median=float(np.median(r)), rms=float(np.sqrt((r ** 2).mean())),
                    p95=float(np.percentile(r, 95)))

    H_raw, inl = mag(pts_t, pts_s)
    rs_raw = rs(pts_t[inl], pts_s[inl], H_raw)
    log(f"raw: 内点 {inl.sum()} ({inl.mean():.1%}), resid med {rs_raw['median']:.2f}px")

    pr, nr = refine(tpl, smp, pts_t, pts_s)
    ok = nr >= NCC_MIN
    H_ref, inl_r = mag(pts_t[ok], pr[ok])
    rs_ref = rs(pts_t[ok][inl_r], pr[ok][inl_r], H_ref)
    log(f"refined: 内点 {inl_r.sum()} ({inl_r.mean():.1%}), resid med {rs_ref['median']:.2f}px")

    res = dict(kpts=[int(f_t['keypoints'].shape[1]), int(f_s['keypoints'].shape[1])],
               matches=int(len(m)), extract_s=round(t_ext, 1), match_s=round(t_match, 2),
               raw=dict(inliers=int(inl.sum()), ratio=float(inl.mean()), resid=rs_raw),
               refined=dict(ncc_keep=int(ok.sum()), inliers=int(inl_r.sum()),
                            ratio=float(inl_r.mean()), resid=rs_ref))
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    lines = [f"SuperPoint+LightGlue @0.5x kpts={MAX_KPTS}",
             f"kpts {res['kpts'][0]}/{res['kpts'][1]}, LG matches {res['matches']}, 提取 {t_ext:.1f}s 匹配 {t_match:.2f}s",
             f"raw: 内点 {inl.sum()} ({inl.mean():.1%}) med/rms/p95 {rs_raw['median']:.2f}/{rs_raw['rms']:.2f}/{rs_raw['p95']:.2f}",
             f"refined: 内点 {inl_r.sum()} ({inl_r.mean():.1%}) med/rms/p95 {rs_ref['median']:.2f}/{rs_ref['rms']:.2f}/{rs_ref['p95']:.2f}"]
    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # 匹配连线图
    s = 1400.0 / Hf
    both = np.full((Hf, Wf * 2 + 8), 255, np.uint8)
    both[:, :Wf] = tpl
    both[:, Wf + 8:] = smp
    ov = cv2.cvtColor(cv2.resize(both, None, fx=s, fy=s), cv2.COLOR_GRAY2BGR)
    idx = np.linspace(0, len(pts_t) - 1, min(500, len(pts_t))).astype(int)
    for i in idx:
        if not inl[i]:
            continue
        p1 = (int(pts_t[i, 0] * s), int(pts_t[i, 1] * s))
        p2 = (int((pts_s[i, 0] + Wf + 8) * s), int(pts_s[i, 1] * s))
        cv2.line(ov, p1, p2, (0, 170, 0), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(OUT_DIR, "superpoint_matches.png"), ov)


if __name__ == "__main__":
    main()

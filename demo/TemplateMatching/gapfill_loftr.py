# -*- coding: utf-8 -*-
"""LoFTR（kornia, detector-free Transformer 直接匹配）真实对评测。输出 gapfill_loftr_output/"""
import cv2
import numpy as np
import os, time, json, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from classical_feature_test import refine, mag_h, resid_stats

TEMPLATE_PATH = r"D:\Users\Administrator\Desktop\weights222\object_region_full_res.jpg"
SAMPLE_PATH = r"D:\Users\Administrator\Desktop\weights222\corresponding_sample_region_full_res.jpg"
OUT_DIR = os.path.join(HERE, "gapfill_loftr_output")
os.makedirs(OUT_DIR, exist_ok=True)
DOWNSCALE = 0.5
LONG_SIDE = 1600   # LoFTR 输入尺寸（CPU 限制），关键点坐标按比例映射回半分辨率


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    log("读取图像…")
    tpl0 = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
    smp0 = cv2.imread(SAMPLE_PATH, cv2.IMREAD_GRAYSCALE)
    Wf, Hf = int(tpl0.shape[1] * DOWNSCALE), int(tpl0.shape[0] * DOWNSCALE)
    tpl = cv2.resize(tpl0, (Wf, Hf), interpolation=cv2.INTER_AREA)
    smp = cv2.resize(smp0, (Wf, Hf), interpolation=cv2.INTER_AREA)

    # LoFTR 输入（最长边 LONG_SIDE，8 的倍数）
    sc = LONG_SIDE / max(Wf, Hf)
    lw, lh = int(Wf * sc) // 8 * 8, int(Hf * sc) // 8 * 8
    t_l = cv2.resize(tpl, (lw, lh), interpolation=cv2.INTER_AREA)
    s_l = cv2.resize(smp, (lw, lh), interpolation=cv2.INTER_AREA)
    log(f"LoFTR 输入 {lw}x{lh} (缩放 {sc:.3f})")

    import torch
    import kornia as K
    from kornia.feature import LoFTR
    matcher = LoFTR(pretrained="outdoor").eval()
    img0 = torch.from_numpy(t_l.astype(np.float32) / 255.0)[None, None]
    img1 = torch.from_numpy(s_l.astype(np.float32) / 255.0)[None, None]

    t0 = time.time()
    with torch.inference_mode():
        resp = matcher({"image0": img0, "image1": img1})
    t_match = time.time() - t0
    log(f"LoFTR 输出 keys: {list(resp.keys())}, 耗时 {t_match:.1f}s")
    kp0 = resp["keypoints0"].cpu().numpy()
    kp1 = resp["keypoints1"].cpu().numpy()
    conf = resp.get("confidence", resp.get("match_confidence"))
    if conf is not None:
        conf = conf.cpu().numpy()
        mask = conf >= 0.5
        kp0, kp1, conf = kp0[mask], kp1[mask], conf[mask]
        log(f"置信度>=0.5 过滤后 {len(kp0)} 对")
    # 映射回半分辨率
    sx, sy = Wf / lw, Hf / lh
    pts_t = np.column_stack([kp0[:, 0] * sx, kp0[:, 1] * sy])
    pts_s = np.column_stack([kp1[:, 0] * sx, kp1[:, 1] * sy])

    H_raw, inl = mag_h(pts_t, pts_s)
    rs_raw = resid_stats(pts_t[inl], pts_s[inl], H_raw)
    log(f"raw: 内点 {inl.sum()} ({inl.mean():.1%}), resid med {rs_raw['median']:.2f}px")

    pr, nr = refine(tpl, smp, pts_t, pts_s)
    ok = nr >= 0.5
    H_ref, inl_r = mag_h(pts_t[ok], pr[ok])
    rs_ref = resid_stats(pts_t[ok][inl_r], pr[ok][inl_r], H_ref)
    log(f"refined: 内点 {inl_r.sum()} ({inl_r.mean():.1%}), resid med {rs_ref['median']:.2f}px")

    res = dict(matches=int(len(pts_t)), t_match=round(t_match, 1),
               raw=dict(inliers=int(inl.sum()), ratio=float(inl.mean()), resid=rs_raw),
               refined=dict(ncc_keep=int(ok.sum()), inliers=int(inl_r.sum()),
                            ratio=float(inl_r.mean()), resid=rs_ref))
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    lines = [f"LoFTR outdoor @ 输入{lw}x{lh}",
             f"匹配 {res['matches']} 对, 推理 {t_match:.1f}s",
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
    cv2.imwrite(os.path.join(OUT_DIR, "loftr_matches.png"), ov)


if __name__ == "__main__":
    main()

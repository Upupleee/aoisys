# -*- coding: utf-8 -*-
"""
形状匹配做初值 + NCC 稠密 + TPS 的管线 vs 纯 NCC 管线（真实对，同口径对比）。
分支A（基准）: NCC pass1 全窗 -> H -> pass2 锚定 -> pass3 小patch -> H(+TPS)
分支B（实验）: 形状匹配(真值角±2°热启动) -> 刚性初值 H_init -> NCC 锚定稠密 -> H(+TPS)
对比：最终 H 的稠密偏差、匹配内点残差、TPS holdout、总耗时。
"""
import cv2
import numpy as np
import os, time, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import synthetic_tps_validation as stv
from shape_match_test import rot_about_center
from preprocess_shape_test import shape_search, make_variants

OUT_DIR = os.path.join(HERE, "shape_tps_output")
os.makedirs(OUT_DIR, exist_ok=True)
DOWNSCALE = 0.5


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def dense_dev(Ha, Hb, W, H, stride=8):
    xs = np.arange(0, W, stride, dtype=np.float64)
    ys = np.arange(0, H, stride, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    t = np.stack([X.ravel(), Y.ravel()], axis=-1)
    a = np.c_[t, np.ones(len(t))] @ Ha.T
    b = np.c_[t, np.ones(len(t))] @ Hb.T
    d = np.linalg.norm(a[:, :2] / a[:, 2:3] - b[:, :2] / b[:, 2:3], axis=1)
    return dict(median=float(np.median(d)), rms=float(np.sqrt(np.mean(d ** 2))),
                p95=float(np.percentile(d, 95)), mx=float(d.max()))


def main():
    log("读取图像…")
    tpl0 = cv2.imread(stv.TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
    smp0 = cv2.imread(r"D:\Users\Administrator\Desktop\weights222\corresponding_sample_region_full_res.jpg",
                      cv2.IMREAD_GRAYSCALE)
    Wf, Hf = int(tpl0.shape[1] * DOWNSCALE), int(tpl0.shape[0] * DOWNSCALE)
    tpl = cv2.resize(tpl0, (Wf, Hf), interpolation=cv2.INTER_AREA)
    smp = cv2.resize(smp0, (Wf, Hf), interpolation=cv2.INTER_AREA)

    def inlier_resid(pts, disp, Hm, thres=6.0):
        Hm2, mask = cv2.findHomography(pts.reshape(-1, 1, 2).astype(np.float32),
                                       (pts + disp).reshape(-1, 1, 2).astype(np.float32),
                                       cv2.USAC_MAGSAC, thres, maxIters=5000)
        r = np.linalg.norm(pts[mask.ravel().astype(bool)] + disp[mask.ravel().astype(bool)] -
                           cv2.perspectiveTransform(pts[mask.ravel().astype(bool)].reshape(-1, 1, 2).astype(np.float32),
                                                    Hm2).reshape(-1, 2), axis=1)
        return Hm2, float(np.sqrt((r ** 2).mean())), float(mask.mean()), r

    # ---------- 分支A: 纯 NCC 三遍 + TPS ----------
    log("== 分支A: 纯 NCC 管线 ==")
    tA = time.time()
    pts, disp, sc = stv.match_grid(tpl, smp)
    keep = stv.consistency_filter(pts, disp, sc)
    pts, disp = pts[keep], disp[keep]
    HA, rA, ratioA, _ = inlier_resid(pts, disp, None)
    pts2, disp2, sc2 = stv.match_grid(tpl, smp, H0=HA, search_r=48)
    keep2 = stv.consistency_filter(pts2, disp2, sc2)
    pts2, disp2 = pts2[keep2], disp2[keep2]
    HA, rA, ratioA, _ = inlier_resid(pts2, disp2, HA)
    pts3, disp3, sc3 = stv.match_grid(tpl, smp, H0=HA, search_r=stv.SMALL_SEARCH, patch=stv.SMALL_PATCH)
    keep3 = stv.consistency_filter(pts3, disp3, sc3, ncc_min=stv.SMALL_NCC)
    pts3, disp3 = pts3[keep3], disp3[keep3]
    HA, rA, ratioA, _ = inlier_resid(pts3, disp3, HA)
    rbfA, lamA, holdA, scaleA, hold_hA = stv.fit_tps(pts3, disp3, HA)
    tA = time.time() - tA
    log(f"A: 点{len(pts3)} 内点残差RMS {rA:.2f}px, TPS holdout {holdA:.2f} vs H {hold_hA:.2f}, t={tA:.1f}s")

    # ---------- 分支B: 形状匹配初值 ----------
    log("== 分支B: 形状匹配初值 ==")
    tB = time.time()
    vt = make_variants(tpl)
    vs = make_variants(smp)
    # 热启动：真值角已知（NCC H），模拟"初值不存在但角度可粗估"的用法
    pts0, disp0, sc0 = stv.match_grid(tpl, smp, search_r=stv.SEARCH_R)
    keep0 = stv.consistency_filter(pts0, disp0, sc0)
    H00, _ = stv.fit_homography(pts0[keep0], disp0[keep0])
    h00, h01, h10, h11 = H00[0, 0], H00[0, 1], H00[1, 0], H00[1, 1]
    ang0 = float(np.degrees(np.arctan2(h10 - h01, h00 + h11)))
    angles = np.arange(ang0 - 2.0, ang0 + 2.01, 0.5)
    out = shape_search(vt["raw"], vs["raw"], angles)
    b_a, b_dy, b_dx, b_s = out[0]
    c = np.array([Wf / 2.0, Hf / 2.0])
    ca, sa = np.cos(np.radians(b_a)), np.sin(np.radians(b_a))
    R = np.array([[ca, -sa], [sa, ca]])
    H_init = np.eye(3)
    H_init[:2, :2] = R
    H_init[:2, 2] = (c + np.array([b_dx, b_dy])) - R @ c
    # 初值与基准 H 的稠密偏差
    dev_init = dense_dev(H_init, HA, Wf, Hf)
    log(f"形状初值: angle={b_a:+.2f} score={b_s:.4f}, 与基准H偏差 med={dev_init['median']:.2f}px RMS={dev_init['rms']:.2f}px")
    t_shape = time.time() - tB

    # 锚定稠密 + TPS（省掉全窗 pass1，这正是形状初值的理论价值）
    tB2 = time.time()
    ptsB, dispB, scB = stv.match_grid(tpl, smp, H0=H_init, search_r=48)
    keepB = stv.consistency_filter(ptsB, dispB, scB)
    ptsB, dispB = ptsB[keepB], dispB[keepB]
    HB, rB, ratioB, _ = inlier_resid(ptsB, dispB, H_init)
    ptsB2, dispB2, scB2 = stv.match_grid(tpl, smp, H0=HB, search_r=stv.SMALL_SEARCH, patch=stv.SMALL_PATCH)
    keepB2 = stv.consistency_filter(ptsB2, dispB2, scB2, ncc_min=stv.SMALL_NCC)
    ptsB2, dispB2 = ptsB2[keepB2], dispB2[keepB2]
    HB, rB, ratioB, _ = inlier_resid(ptsB2, dispB2, HB)
    rbfB, lamB, holdB, scaleB, hold_hB = stv.fit_tps(ptsB2, dispB2, HB)
    tB2 = time.time() - tB2
    tB = time.time() - tB
    log(f"B: 点{len(ptsB2)} 内点残差RMS {rB:.2f}px, TPS holdout {holdB:.2f} vs H {hold_hB:.2f}, "
        f"形状搜索 {t_shape:.1f}s + 稠密 {tB2:.1f}s = {tB:.1f}s")

    dev = dense_dev(HB, HA, Wf, Hf)
    log(f"两分支最终 H 稠密偏差: med={dev['median']:.3f} RMS={dev['rms']:.3f} P95={dev['p95']:.3f}px")

    res = dict(branchA=dict(n=len(pts3), resid_rms=rA, inlier_ratio=ratioA, tps_holdout=holdA,
                            h_holdout=hold_hA, t=round(tA, 1)),
               branchB=dict(n=len(ptsB2), resid_rms=rB, inlier_ratio=ratioB, tps_holdout=holdB,
                            h_holdout=hold_hB, shape_t=round(t_shape, 1), dense_t=round(tB2, 1),
                            t_total=round(tB, 1), init_dev=dev_init, score=b_s, angle=b_a),
               H_dev_A_vs_B=dev)
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    lines = ["形状初值 + NCC + TPS vs 纯 NCC 管线 @0.5x", "",
             f"A 纯NCC : 点{res['branchA']['n']} 残差RMS {rA:.2f}px TPS holdout {holdA:.2f}(H {hold_hA:.2f}) 耗时 {tA:.1f}s",
             f"B 形状初值: 点{res['branchB']['n']} 残差RMS {rB:.2f}px TPS holdout {holdB:.2f}(H {hold_hB:.2f}) "
             f"耗时 {tB:.1f}s (形状 {t_shape:.1f} + 稠密 {tB2:.1f})",
             f"初值偏差 vs 基准H: med {dev_init['median']:.2f}px RMS {dev_init['rms']:.2f}px",
             f"最终 H 偏差 A vs B: med {dev['median']:.3f}px RMS {dev['rms']:.3f}px P95 {dev['p95']:.3f}px"]
    out = "\n".join(lines)
    print("\n" + out)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    main()

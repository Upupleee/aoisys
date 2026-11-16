# -*- coding: utf-8 -*-
"""
补测重跑 2：ECC 改为"NCC pass1 之后精化器"用法（上轮只测了当初始化器，锁死局部最优）。
评估（不依赖 NCC-H 自身当真值）：
  a) 控制点位移残差：|实测 disp - H(t)|，H_ncc vs ECC 精化 H 对比；
  b) 后配准一致性：按各 H 翘曲采样图后重跑 NCC 网格匹配，
     残余位移中位数 + NCC 分数中位数——对齐越好残差越接近 0、分数越高。
两种 ECC 配置：全分辨率单层（初值已近，便宜）与 2→1 两层金字塔。
输出：retest_ecc_refiner_output/
"""
import cv2
import numpy as np
import os, time, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import synthetic_tps_validation as stv

OUT_DIR = os.path.join(HERE, "retest_ecc_refiner_output")
os.makedirs(OUT_DIR, exist_ok=True)
DOWNSCALE = 0.5


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def apply_h(Hm, pts):
    w = np.c_[pts, np.ones(len(pts))] @ Hm.T
    return w[:, :2] / w[:, 2:3]


def ecc_pyramid(tpl, smp, H_init, levels):
    """手工金字塔 ECC，返回 (H, cc, 耗时)。"""
    Wf, Hf = tpl.shape[1], tpl.shape[0]
    Hm = H_init.copy()
    cc = -1.0
    for f in levels:
        tw = cv2.resize(tpl, (Wf // f, Hf // f), interpolation=cv2.INTER_AREA)
        sw = cv2.resize(smp, (Wf // f, Hf // f), interpolation=cv2.INTER_AREA)
        S = np.diag([1.0 / f, 1.0 / f, 1.0])
        w = S @ Hm @ np.diag([f, f, 1.0])
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-7)
        try:
            cc, w = cv2.findTransformECC(tw.astype(np.float32), sw.astype(np.float32),
                                         w.astype(np.float32).copy(), cv2.MOTION_HOMOGRAPHY, crit, None, 5)
        except cv2.error as e:
            log(f"  ECC 1/{f} 失败: {e}")
            return None, -1.0, 0.0
        Hm = np.diag([f, f, 1.0]) @ w.astype(np.float64) @ np.diag([1.0 / f, 1.0 / f, 1.0])
        log(f"  ECC 1/{f}: cc={cc:.5f}")
    return Hm, cc, 0.0


def ctrl_resid(pts_t, disp, Hm):
    """控制点位移残差：|t+disp - H(t)|。"""
    pred = apply_h(Hm, pts_t)
    r = np.linalg.norm(pts_t + disp - pred, axis=1)
    return dict(median=float(np.median(r)), rms=float(np.sqrt((r ** 2).mean())),
                p95=float(np.percentile(r, 95)))


def postwarp_check(tpl, smp, Hm, tag):
    """按 H 翘曲 smp 到模板坐标系后重跑 NCC 网格匹配，看残余位移与分数。"""
    Wf, Hf = tpl.shape[1], tpl.shape[0]
    warped = cv2.warpPerspective(smp, Hm, (Wf, Hf), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    pts, disp, sc = stv.match_grid(tpl, warped)
    d = np.linalg.norm(disp, axis=1)
    res = dict(med_resid=float(np.median(d)), p95_resid=float(np.percentile(d, 95)),
               med_ncc=float(np.median(sc)), n=int(len(d)))
    log(f"  {tag}: 重匹配 {res['n']} 点, 残余位移 med={res['med_resid']:.3f} P95={res['p95_resid']:.3f}px, "
        f"NCC med={res['med_ncc']:.4f}")
    return res


def main():
    log("读取图像…")
    tpl0 = cv2.imread(stv.TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
    smp0 = cv2.imread(r"D:\Users\Administrator\Desktop\weights222\corresponding_sample_region_full_res.jpg",
                      cv2.IMREAD_GRAYSCALE)
    Wf, Hf = int(tpl0.shape[1] * DOWNSCALE), int(tpl0.shape[0] * DOWNSCALE)
    tpl = cv2.resize(tpl0, (Wf, Hf), interpolation=cv2.INTER_AREA)
    smp = cv2.resize(smp0, (Wf, Hf), interpolation=cv2.INTER_AREA)

    pts, disp, sc = stv.match_grid(tpl, smp)
    keep = stv.consistency_filter(pts, disp, sc)
    H_ncc, _ = stv.fit_homography(pts[keep], disp[keep])
    n_keep = int(keep.sum())
    log(f"NCC 控制点 {n_keep}")
    results = {"n_ctrl": n_keep}

    # 基线
    results["H_ncc"] = dict(ctrl=ctrl_resid(pts[keep], disp[keep], H_ncc))
    log(f"H_ncc 控制点残差 med={results['H_ncc']['ctrl']['median']:.3f}px")

    log("== 后配准基线（H_ncc 翘曲后重匹配）==")
    t0 = time.time()
    results["H_ncc"]["postwarp"] = postwarp_check(tpl, smp, H_ncc, "H_ncc")
    log(f"  ({time.time() - t0:.1f}s)")

    # ECC 精化：单层 vs 两层
    for tag, levels in [("ecc_1level", (1,)), ("ecc_pyr21", (2, 1))]:
        log(f"== ECC 精化 {tag} (levels={levels}) ==")
        t0 = time.time()
        Hm, cc, _ = ecc_pyramid(tpl, smp, H_ncc, levels)
        dt = time.time() - t0
        if Hm is None:
            results[tag] = dict(failed=True, t=round(dt, 1))
            continue
        xs = np.arange(0, Wf, 64, dtype=np.float64)
        ys = np.arange(0, Hf, 64, dtype=np.float64)
        X, Y = np.meshgrid(xs, ys)
        grid = np.stack([X.ravel(), Y.ravel()], axis=-1)
        dev = np.linalg.norm(apply_h(Hm, grid) - apply_h(H_ncc, grid), axis=1)
        results[tag] = dict(cc=float(cc), t=round(dt, 1),
                            ctrl=ctrl_resid(pts[keep], disp[keep], Hm),
                            H_move=dict(median=float(np.median(dev)), mx=float(dev.max())))
        log(f"{tag}: cc={cc:.5f}, 控制点残差 med={results[tag]['ctrl']['median']:.3f}px, "
            f"H 位移 med={results[tag]['H_move']['median']:.3f}px, {dt:.1f}s")
        log(f"== 后配准检查（{tag} 翘曲后重匹配）==")
        results[tag]["postwarp"] = postwarp_check(tpl, smp, Hm, tag)

    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    lines = [f"控制点 {n_keep}",
             f"H_ncc: ctrl med={results['H_ncc']['ctrl']['median']:.3f} RMS={results['H_ncc']['ctrl']['rms']:.3f}px | "
             f"postwarp resid med={results['H_ncc']['postwarp']['med_resid']:.3f}px NCC med={results['H_ncc']['postwarp']['med_ncc']:.4f}"]
    for tag in ("ecc_1level", "ecc_pyr21"):
        if tag in results and not results[tag].get("failed"):
            r = results[tag]
            lines.append(f"{tag}: cc={r['cc']:.4f} {r['t']}s | ctrl med={r['ctrl']['median']:.3f} RMS={r['ctrl']['rms']:.3f}px | "
                         f"postwarp resid med={r['postwarp']['med_resid']:.3f}px NCC med={r['postwarp']['med_ncc']:.4f}")
        else:
            lines.append(f"{tag}: 失败")
    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

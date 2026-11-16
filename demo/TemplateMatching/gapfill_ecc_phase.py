# -*- coding: utf-8 -*-
"""
缺口补测 1/2：ECC（金字塔精化）+ 相位相关/log-polar（全局粗对齐）。
真值：NCC 单应（stv 管线）。指标：稠密偏差 med/RMS、耗时。
输出：gapfill_ecc_phase_output/
"""
import cv2
import numpy as np
import os, time, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import synthetic_tps_validation as stv

OUT_DIR = os.path.join(HERE, "gapfill_ecc_phase_output")
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
    return dict(median=float(np.median(d)), rms=float(np.sqrt((d ** 2).mean())),
                p95=float(np.percentile(d, 95)), mx=float(d.max()))


def apply_h(Hm, p):
    w = Hm @ np.array([p[0], p[1], 1.0])
    return np.array([w[0] / w[2], w[1] / w[2]])


def scale_H(Hm, f):
    """把 H 从 1/f 分辨率映射回原分辨率。"""
    S = np.diag([1.0 / f, 1.0 / f, 1.0])
    return S @ Hm @ np.diag([f, f, 1.0])


def ecc_pyramid(tpl, smp, levels=(8, 4, 2, 1), H_init=np.eye(3), motion=cv2.MOTION_HOMOGRAPHY):
    """手工金字塔 ECC：粗层结果上采样作为细层初值。返回全分辨率 H。"""
    Hm = H_init.copy()
    for f in levels:
        tw = cv2.resize(tpl, (int(tpl.shape[1] / f), int(tpl.shape[0] / f)), interpolation=cv2.INTER_AREA)
        sw = cv2.resize(smp, (int(tpl.shape[1] / f), int(tpl.shape[0] / f)), interpolation=cv2.INTER_AREA)
        w = scale_H(Hm, f)
        warp = w.astype(np.float32).copy()
        tpl_f = tw.astype(np.float32)
        smp_f = sw.astype(np.float32)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-7)
        try:
            cc, warp = cv2.findTransformECC(tpl_f, smp_f, warp, motion, crit, None, 5)
        except cv2.error as e:
            log(f"  ECC level 1/{f} 失败: {e}")
            return None, -1.0
        Hm = scale_H(warp.astype(np.float64), f)
        log(f"  ECC 1/{f}: cc={cc:.4f}")
    return Hm, cc


def main():
    log("读取图像…")
    tpl0 = cv2.imread(stv.TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
    smp0 = cv2.imread(r"D:\Users\Administrator\Desktop\weights222\corresponding_sample_region_full_res.jpg",
                      cv2.IMREAD_GRAYSCALE)
    Wf, Hf = int(tpl0.shape[1] * DOWNSCALE), int(tpl0.shape[0] * DOWNSCALE)
    tpl = cv2.resize(tpl0, (Wf, Hf), interpolation=cv2.INTER_AREA)
    smp = cv2.resize(smp0, (Wf, Hf), interpolation=cv2.INTER_AREA)

    # NCC 真值
    pts, disp, sc = stv.match_grid(tpl, smp)
    keep = stv.consistency_filter(pts, disp, sc)
    H_ncc, m0 = stv.fit_homography(pts[keep], disp[keep])
    h00, h01, h10, h11 = H_ncc[0, 0], H_ncc[0, 1], H_ncc[1, 0], H_ncc[1, 1]
    ang_true = float(np.degrees(np.arctan2(h10 - h01, h00 + h11)))
    log(f"NCC 真值角度 {ang_true:+.3f}°")

    results = {}

    # ---------- 相位相关：平移 ----------
    log("== 相位相关 ==")
    t0 = time.time()
    def hp(g, s=24.0):
        k = int(s * 3) | 1
        return g.astype(np.float64) - cv2.GaussianBlur(g.astype(np.float64), (k, k), s)
    a = hp(tpl) * cv2.createHanningWindow((Wf, Hf), cv2.CV_64F)
    b = hp(smp) * cv2.createHanningWindow((Wf, Hf), cv2.CV_64F)
    pc = cv2.phaseCorrelate(a, b)
    shift = np.array(pc[0])  # (dx, dy)
    t_pc = time.time() - t0
    log(f"相位相关平移 shift={shift}, response={pc[1]:.4f}, {t_pc:.2f}s")
    # 语义校验：与 NCC H 在中心处平移对比
    c0 = apply_h(H_ncc, np.array([Wf / 2, Hf / 2])) - np.array([Wf / 2, Hf / 2])
    log(f"NCC 中心平移 = ({c0[0]:.2f},{c0[1]:.2f})")

    # ---------- log-polar：旋转+缩放 ----------
    t0 = time.time()
    M = cv2.getRotationMatrix2D((Wf / 2, Hf / 2), 0, 1.0)
    M[0, 2] += Wf / 2  # warpPolar 需要 dst 平移
    max_r = min(Wf, Hf) // 2
    pol_t = cv2.warpPolar(tpl.astype(np.float32), (360, 720), (Wf / 2, Hf / 2), max_r,
                          cv2.WARP_POLAR_LOG)
    pol_s = cv2.warpPolar(smp.astype(np.float32), (360, 720), (Wf / 2, Hf / 2), max_r,
                          cv2.WARP_POLAR_LOG)
    pa = hp(pol_t.astype(np.float64), 12.0) * cv2.createHanningWindow((360, 720), cv2.CV_64F)
    pb = hp(pol_s.astype(np.float64), 12.0) * cv2.createHanningWindow((360, 720), cv2.CV_64F)
    pc2 = cv2.phaseCorrelate(pa, pb)
    d_ang = pc2[0][0] * 360.0 / 360.0  # x 轴为角度通道(0..360)
    log_scale = -pc2[0][1] / 720.0 * np.log(max_r)
    scale = float(np.exp(log_scale))
    t_lp = time.time() - t0
    log(f"log-polar: Δangle={d_ang:.3f}°, scale={scale:.5f}, {t_lp:.2f}s (真值角 {ang_true:+.3f}°)")

    # 组合 H：s = R_s · (t - c) + c + shift
    th = np.deg2rad(d_ang)
    R = scale * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    c = np.array([Wf / 2.0, Hf / 2.0])
    H_pc = np.eye(3)
    H_pc[:2, :2] = R
    H_pc[:2, 2] = c + shift - R @ c
    dev_pc = dense_dev(H_pc, H_ncc, Wf, Hf)
    log(f"相位相关组合 H vs NCC: med={dev_pc['median']:.2f} RMS={dev_pc['rms']:.2f} P95={dev_pc['p95']:.2f}px")
    results["phasecorr"] = dict(shift=[float(x) for x in shift], angle=d_ang, scale=scale,
                                dev=dev_pc, t=round(t_pc + t_lp, 2), response=float(pc[1]))

    # ---------- ECC ----------
    log("== ECC 金字塔 ==")
    t0 = time.time()
    H_ecc_id, cc1 = ecc_pyramid(tpl, smp, H_init=np.eye(3))
    t_ecc_id = time.time() - t0
    if H_ecc_id is not None:
        dev_id = dense_dev(H_ecc_id, H_ncc, Wf, Hf)
        log(f"ECC(自举): dev med={dev_id['median']:.2f} RMS={dev_id['rms']:.2f}px, {t_ecc_id:.1f}s")
        results["ecc_identity"] = dict(dev=dev_id, t=round(t_ecc_id, 1), cc=float(cc1))
    else:
        results["ecc_identity"] = dict(failed=True, t=round(t_ecc_id, 1))
        log(f"ECC(自举) 失败, {t_ecc_id:.1f}s")

    t0 = time.time()
    H_ecc_pc, cc2 = ecc_pyramid(tpl, smp, H_init=H_pc)
    t_ecc_pc = time.time() - t0
    if H_ecc_pc is not None:
        dev_pc2 = dense_dev(H_ecc_pc, H_ncc, Wf, Hf)
        log(f"ECC(相位初值): dev med={dev_pc2['median']:.2f} RMS={dev_pc2['rms']:.2f}px, {t_ecc_pc:.1f}s")
        results["ecc_phase_init"] = dict(dev=dev_pc2, t=round(t_ecc_pc, 1), cc=float(cc2))
    else:
        results["ecc_phase_init"] = dict(failed=True, t=round(t_ecc_pc, 1))
        log(f"ECC(相位初值) 失败, {t_ecc_pc:.1f}s")

    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(dict(ang_true=ang_true, results=results), f, indent=2, ensure_ascii=False)

    # 汇总图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    if "dev" in results.get("phasecorr", {}):
        rows.append(("PhaseCorr+logpolar", results["phasecorr"]["dev"]))
    if "dev" in results.get("ecc_identity", {}):
        rows.append(("ECC(identity)", results["ecc_identity"]["dev"]))
    if "dev" in results.get("ecc_phase_init", {}):
        rows.append(("ECC(phase init)", results["ecc_phase_init"]["dev"]))
    if rows:
        names = [r[0] for r in rows]
        x = np.arange(len(rows))
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.bar(x - 0.18, [r[1]["median"] for r in rows], 0.36, label="median dev (px)")
        ax.bar(x + 0.18, [r[1]["rms"] for r in rows], 0.36, label="RMS dev (px)")
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel("dense dev vs NCC-H (px @0.5x)")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, axis="y"); ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "ecc_phase_dev.png"), dpi=130)
        plt.close(fig)
    lines = [f"真值角度 {ang_true:+.3f}°",
             f"phasecorr: shift={results['phasecorr'].get('shift')}, angle={results['phasecorr'].get('angle')}, "
             f"dev med={results['phasecorr'].get('dev', {}).get('median', 'NA')}",
             f"ecc_identity: {'failed' if results['ecc_identity'].get('failed') else results['ecc_identity']['dev']}",
             f"ecc_phase_init: {'failed' if results['ecc_phase_init'].get('failed') else results['ecc_phase_init']['dev']}"]
    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

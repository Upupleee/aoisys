# -*- coding: utf-8 -*-
"""
缺口补测 2/2：Farnebäck 稠密光流作为形变场直接估计器（不经稀疏点+TPS）。
评估：光流位移场 vs NCC 单应位移场的稠密偏差；光流 vs NCC 控制点实测位移的一致性。
输出：gapfill_farneback_output/
"""
import cv2
import numpy as np
import os, time, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import synthetic_tps_validation as stv

OUT_DIR = os.path.join(HERE, "gapfill_farneback_output")
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


def flow_to_bgr(flow, step=24):
    """HSV 色轮可视化（下采样箭头）。"""
    h, w = flow.shape[:2]
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    m = mag.max() + 1e-9
    hsv[..., 2] = np.clip(mag / m * 255, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def main():
    log("读取图像…")
    tpl0 = cv2.imread(stv.TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
    smp0 = cv2.imread(r"D:\Users\Administrator\Desktop\weights222\corresponding_sample_region_full_res.jpg",
                      cv2.IMREAD_GRAYSCALE)
    Wf, Hf = int(tpl0.shape[1] * DOWNSCALE), int(tpl0.shape[0] * DOWNSCALE)
    tpl = cv2.resize(tpl0, (Wf, Hf), interpolation=cv2.INTER_AREA)
    smp = cv2.resize(smp0, (Wf, Hf), interpolation=cv2.INTER_AREA)

    # NCC 真值 H + 控制点实测位移
    pts, disp, sc = stv.match_grid(tpl, smp)
    keep = stv.consistency_filter(pts, disp, sc)
    H_ncc, _ = stv.fit_homography(pts[keep], disp[keep])
    log(f"NCC 控制点 {int(keep.sum())}")

    # Farnebäck 稠密光流：prev=tpl, next=smp → flow(t) = t 在采样图中的位置偏移
    log("计算 Farnebäck 光流…")
    t0 = time.time()
    flow = cv2.calcOpticalFlowFarneback(
        tpl, smp, None, 0.5, 6, 41, 10, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    t_flow = time.time() - t0
    log(f"光流完成 {flow.shape}, {t_flow:.1f}s")

    # 1) 光流 vs NCC 单应（稠密网格）
    xs = np.arange(0, Wf, 8, dtype=np.float64)
    ys = np.arange(0, Hf, 8, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    t = np.stack([X.ravel(), Y.ravel()], axis=-1)
    th = np.c_[t, np.ones(len(t))] @ H_ncc.T
    tgt = th[:, :2] / th[:, 2:3]
    fl = flow[::8, ::8].reshape(-1, 2).astype(np.float64)
    dev = np.linalg.norm(fl - (tgt - t), axis=1)
    dev = dev[np.isfinite(dev)]
    stats = dict(median=float(np.median(dev)), rms=float(np.sqrt((dev ** 2).mean())),
                 p95=float(np.percentile(dev, 95)), mx=float(dev.max()))
    log(f"光流 vs NCC-H 稠密偏差: med={stats['median']:.2f} RMS={stats['rms']:.2f} P95={stats['p95']:.2f}px")

    # 2) 光流 vs 控制点实测位移（一致性）
    gi = np.clip(pts[keep].astype(int), 0, np.array([Wf - 1, Hf - 1]))
    fl_pts = flow[gi[:, 1], gi[:, 0]]
    agree = np.linalg.norm(fl_pts - disp[keep], axis=1)
    agree = agree[np.isfinite(agree)]
    agree_stats = dict(median=float(np.median(agree)), rms=float(np.sqrt((agree ** 2).mean())),
                       p95=float(np.percentile(agree, 95)))
    log(f"光流 vs 控制点实测位移: med={agree_stats['median']:.2f} RMS={agree_stats['rms']:.2f} P95={agree_stats['p95']:.2f}px")

    res = dict(flow_t=round(t_flow, 1), dev_vs_H=stats, agree_ctrl=agree_stats)
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    # 可视化：光流色轮图 + 一致性热图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 7))
    axes[0].imshow(cv2.cvtColor(cv2.resize(flow_to_bgr(flow), None, fx=0.25, fy=0.25), cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Farnebäck flow (t={t_flow:.0f}s)")
    d2 = np.linalg.norm(fl - (tgt - t), axis=1).reshape(len(ys), len(xs))
    d2 = np.where(np.isfinite(d2), d2, np.nan)
    im = axes[1].imshow(d2, cmap="magma", vmin=0, vmax=np.nanpercentile(d2, 98))
    axes[1].set_title(f"|flow - H(t)-t|  med={stats['median']:.2f}px")
    fig.colorbar(im, ax=axes[1], shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "farneback_flow.png"), dpi=120)
    plt.close(fig)

    lines = [f"Farnebäck: {t_flow:.1f}s",
             f"光流 vs NCC-H 稠密偏差: med={stats['median']:.2f} RMS={stats['rms']:.2f} P95={stats['p95']:.2f} max={stats['mx']:.2f} px@0.5x",
             f"光流 vs 控制点实测位移: med={agree_stats['median']:.2f} RMS={agree_stats['rms']:.2f} P95={agree_stats['p95']:.2f} px@0.5x"]
    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

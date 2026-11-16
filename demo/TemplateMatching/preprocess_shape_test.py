# -*- coding: utf-8 -*-
"""
预处理 x 形状匹配（Steger风格倍角方向场）对照实验。
以 NCC+MAGSAC 单应为真值，在真值角 ±2° 范围内做形状匹配搜索（各变体同口径），
比较各预处理变体的：峰值 score、位置/角度误差、耗时。
"""
import cv2
import numpy as np
import os, time, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import synthetic_tps_validation as stv
from shape_match_test import preprocess, orient_field, rot_about_center, unwrap

stv.STRIDE = 256
TEMPLATE_PATH = stv.TEMPLATE_PATH
SAMPLE_PATH = r"D:\Users\Administrator\Desktop\weights222\corresponding_sample_region_full_res.jpg"
OUT_DIR = os.path.join(HERE, "preprocess_shape_output")
os.makedirs(OUT_DIR, exist_ok=True)
DOWNSCALE = 0.5


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def zn(img):
    return (img - img.mean()) / max(img.std(), 1e-9)


def make_variants(img):
    g = img.astype(np.float32)
    v = {"raw": g, "zscore": zn(g)}
    clahe = cv2.createCLAHE(2.0, (8, 8)).apply(img)
    v["clahe"] = clahe.astype(np.float32)
    thr, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    v["otsu_bin"] = np.where(g > thr, 255.0, 0.0).astype(np.float32)
    ab = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 10)
    v["adaptive_bin"] = ab.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
    v["sobel_mag"] = np.sqrt(gx * gx + gy * gy)
    return v


def shape_search(tpl_v, smp_v, angles, edge_pct=97.0):
    """单尺度 FFT 倍角方向场搜索，返回按 score 降序 (angle, dy, dx, score)。"""
    smp_hp = preprocess(smp_v)
    Gs, ns = orient_field(smp_hp, edge_pct)
    Fs = np.fft.fft2(Gs)
    Fs[0, 0] = 0
    out = []
    for a in angles:
        Gt, nt = orient_field(preprocess(rot_about_center(tpl_v, a)), edge_pct)
        F = np.fft.fft2(Gt)
        F[0, 0] = 0
        corr = np.fft.ifft2(F * np.conj(Fs)).real / max(ns, 1)
        py, px = np.unravel_index(np.argmax(corr), corr.shape)
        out.append((float(a), unwrap(py, corr.shape[0]), unwrap(px, corr.shape[1]),
                    float(corr[py, px])))
    out.sort(key=lambda t: -t[3])
    return out


def main():
    log("读取图像…")
    tpl0 = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
    smp0 = cv2.imread(SAMPLE_PATH, cv2.IMREAD_GRAYSCALE)
    Wf, Hf = int(tpl0.shape[1] * DOWNSCALE), int(tpl0.shape[0] * DOWNSCALE)
    tpl8 = cv2.resize(tpl0, (Wf, Hf), interpolation=cv2.INTER_AREA)
    smp8 = cv2.resize(smp0, (Wf, Hf), interpolation=cv2.INTER_AREA)

    # NCC 真值单应
    log("拟合 NCC 真值单应…")
    pts, disp, sc = stv.match_grid(tpl8, smp8)
    keep = stv.consistency_filter(pts, disp, sc)
    pts, disp = pts[keep], disp[keep]
    H0, m0 = stv.fit_homography(pts, disp)
    h00, h01, h10, h11 = H0[0, 0], H0[0, 1], H0[1, 0], H0[1, 1]
    ang_true = float(np.degrees(np.arctan2(h10 - h01, h00 + h11)))
    log(f"真值角度 {ang_true:+.3f}°, 内点率 {m0.mean():.1%}")

    angles = np.arange(ang_true - 2.0, ang_true + 2.01, 0.5)
    vt = make_variants(tpl8)
    vs = make_variants(smp8)

    # 评估网格（半分辨率坐标）
    xs = np.arange(0, Wf, 128, dtype=np.float64)
    ys = np.arange(0, Hf, 128, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    t = np.stack([X.ravel(), Y.ravel()], axis=-1)
    th = np.c_[t, np.ones(len(t))] @ H0.T
    tgt = th[:, :2] / th[:, 2:3]

    results = []
    for name in vt:
        t0 = time.time()
        out = shape_search(vt[name], vs[name], angles)
        dt = time.time() - t0
        b_a, b_dy, b_dx, b_s = out[0]
        # 刚性模型误差 vs 真值 H：p_s = R(p_t - c) + c + d
        c = np.array([Wf / 2.0, Hf / 2.0])
        ca, sa = np.cos(np.radians(b_a)), np.sin(np.radians(b_a))
        R = np.array([[ca, -sa], [sa, ca]])
        pred = (t - c) @ R.T + c + np.array([b_dx, b_dy])
        err = np.linalg.norm(pred - tgt, axis=1)
        results.append(dict(name=name, score=b_s, angle=b_a, angle_err=abs(b_a - ang_true),
                            err_median=float(np.median(err)), err_rms=float(np.sqrt((err ** 2).mean())),
                            err_p95=float(np.percentile(err, 95)), t=round(dt, 1)))
        log(f"{name:<12} score={b_s:.4f} angle={b_a:+.2f}(Δ{abs(b_a-ang_true):.2f}°) "
            f"err med/RMS={np.median(err):.2f}/{np.sqrt((err**2).mean()):.2f}px t={dt:.1f}s")

    lines = ["预处理 x 形状匹配对照 @0.5x（真值=NCC单应; 搜索=真值角±2°步0.5°, 内部高通σ8照旧）", ""]
    lines.append(f"{'变体':<13}{'score':>8}{'角度':>8}{'Δ角度':>8}{'err_med':>9}{'err_RMS':>9}{'err_P95':>9}{'耗时s':>7}")
    for r in results:
        lines.append(f"{r['name']:<13}{r['score']:>8.4f}{r['angle']:>8.2f}{r['angle_err']:>8.2f}"
                     f"{r['err_median']:>9.2f}{r['err_rms']:>9.2f}{r['err_p95']:>9.2f}{r['t']:>7.1f}")
    out = "\n".join(lines)
    print("\n" + out)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(out + "\n")
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(dict(ang_true=ang_true, results=results), f, indent=2, ensure_ascii=False)

    # 柱状对比图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [r["name"] for r in results]
    x = np.arange(len(names))
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    ax[0].bar(x, [r["score"] for r in results], color="#4a7dbd")
    ax[0].set_title("peak score")
    ax[1].bar(x, [r["err_rms"] for r in results], color="#c0504d")
    ax[1].set_title("position err RMS (px)")
    ax[2].bar(x, [r["t"] for r in results], color="#9bbb59")
    ax[2].set_title("search time (s)")
    for a in ax:
        a.set_xticks(x); a.set_xticklabels(names, rotation=30, fontsize=8); a.grid(alpha=0.3, axis="y")
    fig.suptitle(f"Shape matching vs preprocessing (truth angle {ang_true:+.2f} deg)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "shape_preprocess_compare.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()

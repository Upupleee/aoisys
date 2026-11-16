# -*- coding: utf-8 -*-
"""
预处理对照实验：灰度差异对 匹配质量 + 对齐后差异比对 的影响。
变体：raw / zscore / highpass / CLAHE / Otsu二值 / 自适应二值 / Sobel幅值
- 匹配质量：同一 NCC 网格管线（pass1+过滤+MAGSAC）的内点率与内点残差
- 差异比对：用 raw 变体拟合的单应（统一对齐基准）校正后，各变体的 z-norm 差分 RMS/P99
"""
import cv2
import numpy as np
import os, time, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import synthetic_tps_validation as stv  # 复用 match_grid / consistency_filter / fit_homography

stv.STRIDE = 256          # 本实验只需 pass1，网格放粗提效
TEMPLATE_PATH = stv.TEMPLATE_PATH
SAMPLE_PATH = r"D:\Users\Administrator\Desktop\weights222\corresponding_sample_region_full_res.jpg"
OUT_DIR = os.path.join(HERE, "preprocess_output")
os.makedirs(OUT_DIR, exist_ok=True)
DOWNSCALE = 0.5
BORDER = 48               # 差分评估时剔除的边框宽度(@0.5x)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def zn(img, mask=None):
    if mask is None:
        return (img - img.mean()) / max(img.std(), 1e-9)
    v = img[mask]
    return (img - v.mean()) / max(v.std(), 1e-9)


def make_variants(img):
    """返回 dict[name, float32]，全部为匹配友好的灰度表达。"""
    g = img.astype(np.float32)
    v = {}
    v["raw"] = g
    v["zscore"] = zn(g)
    hp = g - cv2.GaussianBlur(g, (0, 0), 8.0)
    v["highpass"] = zn(hp)
    clahe = cv2.createCLAHE(2.0, (8, 8)).apply(img)
    v["clahe"] = clahe.astype(np.float32)
    thr, ob = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    v["otsu_bin"] = np.where(g > thr, 255.0, 0.0).astype(np.float32)
    ab = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 51, 10)
    v["adaptive_bin"] = ab.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
    v["sobel_mag"] = np.sqrt(gx * gx + gy * gy)
    log(f"Otsu 阈值 = {thr:.0f}")
    return v


def main():
    log("读取图像…")
    tpl0 = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
    smp0 = cv2.imread(SAMPLE_PATH, cv2.IMREAD_GRAYSCALE)
    Wf, Hf = int(tpl0.shape[1] * DOWNSCALE), int(tpl0.shape[0] * DOWNSCALE)
    tpl8 = cv2.resize(tpl0, (Wf, Hf), interpolation=cv2.INTER_AREA)
    smp8 = cv2.resize(smp0, (Wf, Hf), interpolation=cv2.INTER_AREA)

    vt = make_variants(tpl8)
    vs = make_variants(smp8)
    names = list(vt.keys())

    # 统一对齐基准：raw 变体拟合的 H
    log("拟合基准单应（raw）…")
    pts, disp, sc = stv.match_grid(vt["raw"], vs["raw"])
    keep = stv.consistency_filter(pts, disp, sc)
    pts, disp = pts[keep], disp[keep]
    H0, m0 = stv.fit_homography(pts, disp)
    inl0 = np.linalg.norm(pts[m0] + disp[m0] - cv2.perspectiveTransform(
        pts[m0].reshape(-1, 1, 2).astype(np.float32), H0).reshape(-1, 2), axis=1)
    log(f"基准: 匹配 {len(pts)}, 内点率 {m0.mean():.1%}, 内点残差 RMS {np.sqrt((inl0**2).mean()):.2f}px")

    # 校正采样图到模板系（一次），所有变体共用几何
    xs = np.arange(0, Wf, 2, dtype=np.float64)
    ys = np.arange(0, Hf, 2, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    t = np.stack([X.ravel(), Y.ravel()], axis=-1)
    th = np.c_[t, np.ones(len(t))] @ H0.T
    hm = th[:, :2] / th[:, 2:3]
    rect8 = cv2.remap(smp8, hm[:, 0].reshape(len(ys), len(xs)).astype(np.float32),
                      hm[:, 1].reshape(len(ys), len(xs)).astype(np.float32),
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    tpl_r8 = tpl8[::2, ::2]
    rect_r8 = rect8
    h_r, w_r = tpl_r8.shape
    valid = np.zeros((h_r, w_r), bool)
    valid[BORDER:-BORDER, BORDER:-BORDER] = True

    results = []
    for name in names:
        # 1) 匹配质量
        pts_v, disp_v, sc_v = stv.match_grid(vt[name], vs[name])
        keep_v = stv.consistency_filter(pts_v, disp_v, sc_v)
        pts_v, disp_v = pts_v[keep_v], disp_v[keep_v]
        Hv, mv = stv.fit_homography(pts_v, disp_v)
        rv = np.linalg.norm(pts_v[mv] + disp_v[mv] - cv2.perspectiveTransform(
            pts_v[mv].reshape(-1, 1, 2).astype(np.float32), Hv).reshape(-1, 2), axis=1)
        # 与基准几何的一致性（Hv vs H0 稠密像素差）
        a = np.c_[t, np.ones(len(t))] @ Hv.T
        dH = np.linalg.norm(a[:, :2] / a[:, 2:3] - hm, axis=1)

        # 2) 差异比对（同一几何下，仅预处理不同）
        tv = make_variants(tpl_r8)[name] if False else None  # 占位避免重复构建
        tv = variant_small(vt[name], (w_r, h_r))
        sv = variant_small(vs[name], (w_r, h_r))
        a_n = zn(tv, valid)
        b_n = zn(sv, valid)
        diff = np.abs(a_n - b_n)
        rms_d = float(np.sqrt(np.mean(diff[valid] ** 2)))
        p99_d = float(np.percentile(diff[valid], 99))

        results.append(dict(name=name, n_match=int(len(pts_v)),
                            inlier_ratio=float(mv.mean()),
                            resid_rms=float(np.sqrt((rv ** 2).mean())) if len(rv) else -1,
                            resid_p95=float(np.percentile(rv, 95)) if len(rv) else -1,
                            h_dev_median=float(np.median(dH)), h_dev_p95=float(np.percentile(dH, 95)),
                            diff_rms=rms_d, diff_p99=p99_d))
        log(f"{name:<12} 匹配{len(pts_v):>5} 内点率{mv.mean():>6.1%} 残差RMS{results[-1]['resid_rms']:>6.2f}px "
            f"H偏差med{results[-1]['h_dev_median']:.2f} | 差分RMS{rms_d:.3f} P99{p99_d:.3f}")

    # 汇总
    lines = ["预处理对照实验 @0.5x（对齐几何统一用 raw 变体的 H）", ""]
    lines.append(f"{'变体':<13}{'匹配数':>7}{'内点率':>8}{'残差RMS':>9}{'残差P95':>9}{'H偏差med':>9}{'H偏差P95':>9}{'差分RMS':>9}{'差分P99':>9}")
    for r in results:
        lines.append(f"{r['name']:<13}{r['n_match']:>7}{r['inlier_ratio']:>8.1%}{r['resid_rms']:>9.2f}"
                     f"{r['resid_p95']:>9.2f}{r['h_dev_median']:>9.2f}{r['h_dev_p95']:>9.2f}"
                     f"{r['diff_rms']:>9.3f}{r['diff_p99']:>9.3f}")
    out = "\n".join(lines)
    print("\n" + out)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(out + "\n")
    import json
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 差分可视化（各变体 absdiff 热图拼图）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(results)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, r in zip(axes.ravel(), results):
        tv = variant_small(vt[r["name"]], (w_r, h_r))
        sv = variant_small(vs[r["name"]], (w_r, h_r))
        d = np.abs(zn(tv, valid) - zn(sv, valid))
        d = np.where(valid, d, np.nan)
        im = ax.imshow(d, cmap="magma", vmin=0, vmax=1.5)
        ax.set_title(f"{r['name']}  RMS={r['diff_rms']:.3f}")
        ax.axis("off")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.7)
    fig.suptitle("Aligned absdiff (z-normed) per preprocessing variant")
    fig.savefig(os.path.join(OUT_DIR, "diff_heatmaps.png"), dpi=120)
    plt.close(fig)

    # 变体预览：每个变体的 模板|采样 缩略对比（看预处理效果）
    prev_s = 0.18
    tw, th_ = int(Wf * prev_s), int(Hf * prev_s)
    fig, axes = plt.subplots(n, 2, figsize=(7.2, n * 2.1))
    for i, name in enumerate(names):
        axes[i, 0].imshow(cv2.resize(vt[name], (tw, th_)), cmap="gray")
        axes[i, 1].imshow(cv2.resize(vs[name], (tw, th_)), cmap="gray")
        axes[i, 0].set_ylabel(name, fontsize=9)
        if i == 0:
            axes[i, 0].set_title("template")
            axes[i, 1].set_title("sample")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Preprocessing variants: template | sample")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "variants_preview.png"), dpi=120)
    plt.close(fig)

    # 对齐后品红-绿叠加 zoom（中心 1200x1200 @0.5x）
    ch, cw = h_r // 2, w_r // 2
    half = 600
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, r in zip(axes.ravel(), results):
        tv = variant_small(vt[r["name"]], (w_r, h_r))
        sv = variant_small(vs[r["name"]], (w_r, h_r))
        ovl = np.dstack([sv, tv, sv])  # 模板=绿, 校正采样=品红; 灰白=对齐
        crop = ovl[ch - half:ch + half, cw - half:cw + half].astype(np.uint8)
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{r['name']}  P99={r['diff_p99']:.2f}")
        ax.axis("off")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Aligned magenta-green overlay zoom (gray = aligned)")
    fig.savefig(os.path.join(OUT_DIR, "overlay_zoom.png"), dpi=120)
    plt.close(fig)


def variant_small(img, size):
    """将变体图缩到差分评估尺寸（w/2），保持各变体一致的缩放方式。"""
    w, h = size
    if img.dtype != np.uint8:
        a, b = img.min(), img.max()
        img8 = ((img - a) / max(b - a, 1e-9) * 255).astype(np.uint8)
    else:
        img8 = img
    return cv2.resize(img8, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)


if __name__ == "__main__":
    main()

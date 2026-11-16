# -*- coding: utf-8 -*-
"""
形状匹配（shape-based matching, Steger风格）效果验证 v2
修正：逐像素梯度单位方向向量 + 强边缘掩码（等价 Halcon 打分）+ 高通去光照低频
对比基准：NCC 稠密匹配 + MAGSAC H
"""
import cv2
import numpy as np
import os, time

TEMPLATE_PATH = r"D:\Users\Administrator\Desktop\weights222\object_region_full_res.jpg"
SAMPLE_PATH   = r"D:\Users\Administrator\Desktop\weights222\corresponding_sample_region_full_res.jpg"
OUT_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shape_match_test_output")

COARSE_FACTOR = 4
FINE_FACTOR   = 2
COARSE_ANGLES = np.arange(-12, 12.1, 2.0)
HP_SIGMA      = 8        # 高通 σ（工作尺度），去除光照低频
NCC_HALF, NCC_STEP = 60, 96


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def preprocess(gray):
    """高通去光照低频"""
    sigma = HP_SIGMA
    k = int(sigma * 3) | 1
    return (gray.astype(np.float32) - cv2.GaussianBlur(gray, (k, k), sigma))


def orient_field(img, edge_pct):
    """倍角方向场 exp(i2θ)：极性反转(Δθ=180°)也得满分，等价 Halcon ignore_local_polarity"""
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, 3)
    mag = np.sqrt(gx * gx + gy * gy)
    mask = mag >= np.percentile(mag, edge_pct)
    mag2 = mag[mask] ** 2
    G = np.zeros(img.shape, np.complex128)
    G[mask] = ((gx[mask] ** 2 - gy[mask] ** 2) + 2j * gx[mask] * gy[mask]) / mag2
    return G, int(mask.sum())


def rot_about_center(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def unwrap(v, size):
    """循环相关峰回绕到正确符号"""
    return v - size if v > size / 2 else v


def match_at_scale(tpl_hp, smp_hp, angles, edge_pct_t=97.0, edge_pct_s=97.0):
    """返回按 score 降序的 (angle, dy, dx, score) 列表"""
    Gs, _ = orient_field(smp_hp, edge_pct_s)
    Fs = np.fft.fft2(Gs)
    Fs[0, 0] = 0  # 去DC
    out = []
    for a in angles:
        Gt, nt = orient_field(rot_about_center(tpl_hp, a), edge_pct_t)
        F = np.fft.fft2(Gt)
        F[0, 0] = 0
        corr = np.fft.ifft2(F * np.conj(Fs)).real / max(nt, 1)
        py, px = np.unravel_index(np.argmax(corr), corr.shape)
        out.append((float(a), unwrap(py, corr.shape[0]), unwrap(px, corr.shape[1]),
                    float(corr[py, px])))
    out.sort(key=lambda t: -t[3])
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log("加载图像 ...")
    tpl_bgr = cv2.imread(TEMPLATE_PATH); smp_bgr = cv2.imread(SAMPLE_PATH)
    FH, FW = tpl_bgr.shape[:2]
    tpl_g = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)
    smp_g = cv2.cvtColor(smp_bgr, cv2.COLOR_BGR2GRAY)

    def down(g, f):
        return cv2.resize(g, (FW // f, FH // f), interpolation=cv2.INTER_AREA)

    # ---------- 粗搜索 1/4 ----------
    log(f"形状匹配粗搜索 1/{COARSE_FACTOR}: {COARSE_ANGLES[0]}°~{COARSE_ANGLES[-1]}° ...")
    t0 = time.time()
    tpl_hp_c = preprocess(down(tpl_g, COARSE_FACTOR))
    smp_hp_c = preprocess(down(smp_g, COARSE_FACTOR))
    coarse = match_at_scale(tpl_hp_c, smp_hp_c, COARSE_ANGLES)
    for a, dy, dx, s in coarse:
        log(f"  angle={a:+6.1f}°  d=({dx:5d},{dy:5d})  score={s:.4f}")
    a0, dy0, dx0, s0 = coarse[0]
    t_coarse = time.time() - t0
    log(f"粗搜索: angle={a0:+.1f}° score={s0:.4f} ({t_coarse:.1f}s)")

    # ---------- 精搜索 1/2（最优角附近 ±1° 步进0.5°） ----------
    log(f"精搜索 1/{FINE_FACTOR}: {a0-1:.1f}°~{a0+1:.1f}° ...")
    t0 = time.time()
    tpl_hp_f = preprocess(down(tpl_g, FINE_FACTOR))
    smp_hp_f = preprocess(down(smp_g, FINE_FACTOR))
    fine = match_at_scale(tpl_hp_f, smp_hp_f, np.arange(a0 - 1, a0 + 1.01, 0.5))
    for a, dy, dx, s in fine:
        log(f"  angle={a:+6.2f}°  d=({dx:5d},{dy:5d})  score={s:.4f}")
    b_a, b_dy, b_dx, b_score = fine[0]
    t_fine = time.time() - t0
    log(f"精搜索: angle={b_a:+.2f}° score={b_score:.4f} ({t_fine:.1f}s)")

    # 全分辨率相似变换: p_s = R(p_t - a) + b, 坐标 ×(FW/(FW//FINE_FACTOR))=2
    h, w = smp_hp_f.shape
    c = np.array([w / 2, h / 2])
    ca, sa = np.cos(np.radians(b_a)), np.sin(np.radians(b_a))
    R = np.array([[ca, -sa], [sa, ca]])
    a_pt = c
    b_pt = c + np.array([b_dx, b_dy])
    M_shape = np.eye(2, 3)
    M_shape[:, :2] = R
    M_shape[:, 2] = (b_pt - R @ a_pt) * FINE_FACTOR   # 升全分辨率

    # ---------- NCC 基准 ----------
    log("NCC 基准匹配 ...")
    t0 = time.time()
    tw, sw = down(tpl_g, 2), down(smp_g, 2)
    pts, u, sc = [], [], []
    for ty in range(NCC_HALF + 60, sw.shape[0] - NCC_HALF - 60, NCC_STEP):
        for tx in range(NCC_HALF + 60, sw.shape[1] - NCC_HALF - 60, NCC_STEP):
            r = cv2.matchTemplate(sw[ty-60-NCC_HALF:ty+60+NCC_HALF, tx-60-NCC_HALF:tx+60+NCC_HALF],
                                  tw[ty-NCC_HALF:ty+NCC_HALF, tx-NCC_HALF:tx+NCC_HALF],
                                  cv2.TM_CCOEFF_NORMED)
            _, s, _, loc = cv2.minMaxLoc(r)
            if s > 0.6:
                pts.append((tx, ty)); u.append((loc[0] - 60, loc[1] - 60)); sc.append(s)
    P = np.array(pts, float); U = np.array(u, float)
    from scipy.spatial import cKDTree
    H2, _ = cv2.findHomography(P.astype(np.float32), (P + U).astype(np.float32), cv2.USAC_MAGSAC, 3.0)
    for _ in range(3):
        tree = cKDTree(P); _, nb = tree.query(P, k=12)
        dev = np.linalg.norm(U - np.median(U[nb], axis=1), axis=1)
        g = dev < max(6, 4 * (np.median(dev) + 1e-6))
        if g.all(): break
        P, U = P[g], U[g]
        H2, _ = cv2.findHomography(P.astype(np.float32), (P + U).astype(np.float32), cv2.USAC_MAGSAC, 3.0)
    t_ncc = time.time() - t0
    log(f"NCC 基准: {len(P)} 点 ({t_ncc:.1f}s)")

    S4 = np.diag([2, 2, 1]).astype(np.float64)
    H_full = S4 @ H2.astype(np.float64) @ np.diag([0.5, 0.5, 1])

    # ---------- 一致性评估 ----------
    rng = np.random.default_rng(7)
    sel = rng.choice(len(P), min(800, len(P)), replace=False)
    Pt = P[sel] * 2.0
    Ps = (P + U)[sel] * 2.0
    err_shape = np.linalg.norm(Pt @ M_shape[:, :2].T + M_shape[:, 2] - Ps, axis=1)
    err_h = np.linalg.norm(cv2.perspectiveTransform(Pt.reshape(-1, 1, 2).astype(np.float32),
                                                    H_full).reshape(-1, 2) - Ps, axis=1)
    h00, h01, h10, h11 = H_full[0, 0], H_full[0, 1], H_full[1, 0], H_full[1, 1]
    ang_h = np.degrees(np.arctan2(h10 - h01, h00 + h11))

    log("逐点位置误差(全分辨率px):")
    log(f"  形状匹配: median={np.median(err_shape):.2f} P95={np.percentile(err_shape,95):.2f} RMS={np.sqrt((err_shape**2).mean()):.2f}")
    log(f"  NCC+H  : median={np.median(err_h):.2f} P95={np.percentile(err_h,95):.2f} RMS={np.sqrt((err_h**2).mean()):.2f}")
    log(f"角度: shape={b_a:+.2f}° vs H={ang_h:+.2f}° 差={abs(b_a-ang_h):.2f}°")

    # ---------- 可视化 ----------
    edges = cv2.Canny(cv2.resize(tpl_g, (FW // 2, FH // 2)), 50, 150)
    col = np.zeros((FH // 2, FW // 2, 3), np.uint8); col[..., 1] = edges
    warped_shape = cv2.warpAffine(col, M_shape / 2, (FW // 2, FH // 2))
    warped_h = cv2.warpPerspective(col, H_full @ np.diag([0.5, 0.5, 1]).astype(np.float64),
                                   (FW // 2, FH // 2))
    base = cv2.cvtColor(cv2.resize(smp_g, (FW // 2, FH // 2)), cv2.COLOR_GRAY2BGR)
    both = base.copy()
    both[warped_shape[..., 1] > 0] = (0, 255, 0)
    both[warped_h[..., 1] > 0] = (255, 0, 255)
    zoom = cv2.resize(both, (2800, int(FH / 2 * 2800 / (FW / 2))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", zoom[1200:2600, 800:3600]); buf.tofile(os.path.join(OUT_DIR, "shape_vs_h_zoom.png"))
    ok, buf = cv2.imencode(".png", cv2.resize(both, (1600, int(FH / 2 * 1600 / (FW / 2)))))
    buf.tofile(os.path.join(OUT_DIR, "shape_vs_h_overlay.png"))

    lines = [
        f"形状匹配: 粗搜{t_coarse:.1f}s+精搜{t_fine:.1f}s={t_coarse+t_fine:.1f}s, score={b_score:.4f}, angle={b_a:+.2f}°",
        f"NCC+MAGSAC: {t_ncc:.1f}s, 角度={ang_h:+.2f}°",
        f"逐点误差 median/P95/RMS(px): shape={np.median(err_shape):.2f}/{np.percentile(err_shape,95):.2f}/{np.sqrt((err_shape**2).mean()):.2f}"
        f"  H={np.median(err_h):.2f}/{np.percentile(err_h,95):.2f}/{np.sqrt((err_h**2).mean()):.2f}",
    ]
    print("\n".join(lines))
    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log("完成")


if __name__ == "__main__":
    main()

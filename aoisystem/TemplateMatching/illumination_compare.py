# -*- coding: utf-8 -*-
"""
光照/对比度鲁棒比对实验：NCC+H+TPS 配准后，用 9 种图像表示做差异比对
方法族：0 基线 | 1 空间域(局部z-score/顶帽/Retinex) | 2 频域(高通/同态/梯度域) | 3 结构(LBP/SSIM结构分量)
评估：残差 RMS / P99（越低=光照去除越干净）+ 残差图可视化
"""
import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
import os, time

TEMPLATE_PATH = r"D:\Users\Administrator\Desktop\weights222\object_region_full_res.jpg"
SAMPLE_PATH   = r"D:\Users\Administrator\Desktop\weights222\corresponding_sample_region_full_res.jpg"
OUT_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "illumination_compare_output")

WORK_FACTOR, VIS_WIDTH = 2, 3000
T, D, S_WIN = 160, 96, 60          # NCC tile 参数（半分辨率 px）
TPS_NEIGHBORS, LAMBDA = 64, 0.1


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def gauss(img, sigma):
    k = int(sigma * 3) | 1
    return cv2.GaussianBlur(img, (k, k), sigma)


# ---------- 光照鲁棒表示（返回 float32 残差输入；比对均在该表示域内做 abs 残差） ----------
def repr_raw(I):           # 0 基线：原图灰度
    return I.astype(np.float32)

def repr_zscore(I):        # 1a 局部 z-score
    mu = gauss(I, 25)
    var = gauss((I.astype(np.float32) - mu) ** 2, 25)
    return (I - mu) / (np.sqrt(var) + 1e-3)

def repr_tophat(I):        # 1b 形态学顶帽（估背景去除）
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
    return cv2.morphologyEx(I, cv2.MORPH_TOPHAT, k).astype(np.float32)

def repr_retinex(I):       # 1c 单尺度 Retinex
    return np.log(I.astype(np.float32) + 1) - np.log(gauss(I, 80).astype(np.float32) + 1)

def repr_highpass(I):      # 2a FFT 高斯高通（真·频域实现）
    F = np.fft.rfft2(I.astype(np.float32))
    h, w = I.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    H = 1 - np.exp(-4 * np.pi ** 2 * 25 ** 2 * 0.5 * (fy ** 2 + fx ** 2))  # σ=25 高斯高通
    return np.fft.irfft2(F * H, s=I.shape).astype(np.float32)

def repr_homomorphic(I):   # 2b 同态滤波
    lnI = np.log(I.astype(np.float32) / 255 + 1e-3)
    F = np.fft.rfft2(lnI)
    h, w = I.shape
    fy = np.fft.fftfreq(h)[:, None]; fx = np.fft.rfftfreq(w)[None, :]
    D2 = fy ** 2 + fx ** 2
    D0, c, gL, gH = 0.005, 1.0, 0.5, 2.0
    H = (gH - gL) * (1 - np.exp(-c * D2 / D0 ** 2)) + gL
    return (np.fft.irfft2(F * H, s=I.shape)).astype(np.float32)

def repr_gradient(I):      # 2c 梯度域（返回两通道 gx,gy）
    gx = cv2.Sobel(I, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(I, cv2.CV_32F, 0, 1, 3)
    return np.stack([gx, gy], -1)

def repr_lbp(I):           # 3a LBP 纹理码
    c = I.astype(np.int16)
    out = np.zeros_like(c)
    bit = 0
    for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]:
        sh = np.roll(np.roll(c, dy, 0), dx, 1)
        out |= ((sh > c) << bit).astype(np.int16)
        bit += 1
    return out.astype(np.float32)

def repr_ssim_struct(I):   # 3b SSIM 结构分量（1-s，残差越大结构差异越大）
    If = I.astype(np.float32)
    mu = gauss(If, 11); mu2 = mu * mu
    var = gauss(If * If, 11) - mu2
    var = np.maximum(var, 0)
    sd = np.sqrt(var) + 1e-6
    C3 = 0.03 ** 2
    return None  # 结构分量需要两张图联合计算，见 ssim_struct_residual()


def ssim_struct_residual(A, B):
    """1 - SSIM 结构分量 s=(σxy+C3)/(σxσy+C3)"""
    a = A.astype(np.float32); b = B.astype(np.float32)
    ma = gauss(a, 11); mb = gauss(b, 11)
    va = np.maximum(gauss(a * a, 11) - ma * ma, 0)
    vb = np.maximum(gauss(b * b, 11) - mb * mb, 0)
    cov = gauss(a * b, 11) - ma * mb
    C3 = 0.03 ** 2
    s = (cov + C3) / (np.maximum(np.sqrt(va) * np.sqrt(vb), 1.0) + C3)
    s = np.clip(s, 0.0, 2.0)
    return 1.0 - s


def residual_of(name, A, B):
    """A,B 为同尺度 uint8 灰度图，返回该方法的残差图（float32）"""
    if name == "3b_SSIM结构":
        return ssim_struct_residual(A, B)
    fa, fb = globals()[name](A), globals()[name](B)
    if name == "2c_梯度域":
        d = np.linalg.norm(fa - fb, axis=-1)
        return d / (d.max() + 1e-6) * 255
    if name == "3a_LBP":
        return np.minimum(np.abs(fa - fb), 255)  # 码差（Hamming 的廉价近似）
    lo, hi = np.percentile(fa, [1, 99])
    scale = (hi - lo) / 255 + 1e-6
    return np.abs(fa / scale - fb / scale)


def save_img(path, img):
    """中文/非ASCII路径安全写图"""
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img)
    if ok:
        buf.tofile(path)
    else:
        raise IOError(f"imencode failed: {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log("加载图像 ...")
    tpl_bgr = cv2.imread(TEMPLATE_PATH); smp_bgr = cv2.imread(SAMPLE_PATH)
    FH, FW = tpl_bgr.shape[:2]
    wh, ww = FH // WORK_FACTOR, FW // WORK_FACTOR
    tw = cv2.cvtColor(cv2.resize(tpl_bgr, (ww, wh)), cv2.COLOR_BGR2GRAY)
    sw = cv2.cvtColor(cv2.resize(smp_bgr, (ww, wh)), cv2.COLOR_BGR2GRAY)

    # ---------- NCC 稠密位移场 + 刚性 H + 残差 TPS（与 deformable_tps_test.py 相同） ----------
    log("NCC 稠密匹配 ...")
    half = T // 2
    pts, u, sc = [], [], []
    for ty in range(half + S_WIN, wh - half - S_WIN, D):
        for tx in range(half + S_WIN, ww - half - S_WIN, D):
            r = cv2.matchTemplate(sw[ty-S_WIN-half:ty+S_WIN+half, tx-S_WIN-half:tx+S_WIN+half],
                                  tw[ty-half:ty+half, tx-half:tx+half], cv2.TM_CCOEFF_NORMED)
            _, s, _, loc = cv2.minMaxLoc(r)
            if s > 0.6:
                pts.append((tx, ty)); u.append((loc[0]-S_WIN, loc[1]-S_WIN)); sc.append(s)
    pts = np.array(pts, float); u = np.array(u, float); sc = np.array(sc)
    P, U = pts[sc > 0.6], u[sc > 0.6]
    H, _ = cv2.findHomography(P.astype(np.float32), (P+U).astype(np.float32), cv2.USAC_MAGSAC, 3.0)
    for _ in range(3):
        tree = cKDTree(P); _, nb = tree.query(P, k=12)
        dev = np.linalg.norm(U - np.median(U[nb], axis=1), axis=1)
        g = dev < max(6, 4 * (np.median(dev) + 1e-6))
        if g.all(): break
        P, U = P[g], U[g]
        H, _ = cv2.findHomography(P.astype(np.float32), (P+U).astype(np.float32), cv2.USAC_MAGSAC, 3.0)
    norm = np.array([FW, FH], float)
    f = WORK_FACTOR
    S4 = np.diag([f, f, 1]).astype(np.float64)
    H_full = S4 @ H.astype(np.float64) @ np.diag([1/f, 1/f, 1])
    resid = (P + U) - cv2.perspectiveTransform(P.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
    tps = RBFInterpolator((P * f) / norm, resid * f, kernel='thin_plate_spline',
                          smoothing=LAMBDA, neighbors=TPS_NEIGHBORS)
    log(f"匹配点 {len(P)}, 刚性残差 median={np.median(np.linalg.norm(resid*f,axis=1)):.2f}px")

    # ---------- remap（粗网格 TPS + 放大，加速） ----------
    vw, vh = VIS_WIDTH, int(round(FH * VIS_WIDTH / FW))
    vs = FW / vw
    tpl_vis = cv2.resize(tpl_bgr, (vw, vh), interpolation=cv2.INTER_AREA)
    smp_vis = cv2.resize(smp_bgr, (vw, vh), interpolation=cv2.INTER_AREA)
    g_smp = cv2.cvtColor(smp_vis, cv2.COLOR_BGR2GRAY)

    step = 8
    yg, xg = np.mgrid[0:vh:step, 0:vw:step].astype(np.float64)
    qc = np.stack([xg, yg], -1).reshape(-1, 2) * vs
    Hi = np.linalg.inv(H_full)
    xc = cv2.perspectiveTransform(qc.reshape(-1, 1, 2), Hi).reshape(-1, 2)
    uc = tps(xc / norm)
    x1c = cv2.perspectiveTransform((qc - uc).reshape(-1, 1, 2), Hi).reshape(-1, 2)
    map_small = (x1c / vs).astype(np.float32).reshape(vh // step + (1 if vh % step else 0),
                                                      vw // step + (1 if vw % step else 0), 2)
    map_tps = cv2.resize(map_small, (vw, vh), interpolation=cv2.INTER_CUBIC)
    warped = cv2.remap(tpl_vis, map_tps, None, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    g_tpl = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # 有效掩码：映射坐标在图内（避免模板真黑像素干扰阈值）
    inside = ((map_tps[..., 0] >= 0) & (map_tps[..., 0] <= vw - 1) &
              (map_tps[..., 1] >= 0) & (map_tps[..., 1] <= vh - 1)).astype(np.uint8)
    valid = cv2.erode(inside, np.ones((51, 51), np.uint8)).astype(bool)
    log(f"有效区域: {valid.mean()*100:.1f}%")

    # ---------- 9 种表示的残差比对 ----------
    methods = ["0_基线灰度", "1a_局部zscore", "1b_顶帽", "1c_Retinex",
               "2a_FFT高通", "2b_同态滤波", "2c_梯度域", "3a_LBP", "3b_SSIM结构"]
    fn_map = {"0_基线灰度": repr_raw, "1a_局部zscore": repr_zscore, "1b_顶帽": repr_tophat,
              "1c_Retinex": repr_retinex, "2a_FFT高通": repr_highpass, "2b_同态滤波": repr_homomorphic,
              "2c_梯度域": repr_gradient, "3a_LBP": repr_lbp}
    res_maps, rows = {}, []
    for name in methods:
        t0 = time.time()
        if name == "3b_SSIM结构":
            r = ssim_struct_residual(g_smp, g_tpl)
        else:
            fa, fb = fn_map[name](g_smp), fn_map[name](g_tpl)
            if name == "2c_梯度域":
                d = np.linalg.norm(fa - fb, axis=-1)
                r = d / (d.max() + 1e-6) * 255
            elif name == "3a_LBP":
                r = np.minimum(np.abs(fa - fb), 255)
            else:
                lo, hi = np.percentile(fa, [1, 99])
                scale = (hi - lo) / 255 + 1e-6
                r = np.abs(fa / scale - fb / scale)
        dt = time.time() - t0
        r_v = r[valid]
        rms = float(np.sqrt((r_v ** 2).mean()))
        # 残差动态范围（对缺陷可检性重要：噪声底 vs 峰值对比）
        p99 = float(np.percentile(np.abs(r_v), 99))
        save_img(os.path.join(OUT_DIR, f"residual_{name}.png"),
                    np.clip(np.abs(r) / (p99 + 1e-6) * 255, 0, 255).astype(np.uint8))
        res_maps[name] = np.abs(r) / (p99 + 1e-6)
        rows.append((name, rms, p99, rms / (p99 + 1e-6), dt))
        log(f"{name:14s} RMS={rms:7.2f} P99={p99:7.2f} RMS/P99={rms/(p99+1e-6):.3f} ({dt:.1f}s)")

    # 网格可视化（标签用 ASCII：cv2 内置字体不支持中文）
    ascii_label = {"0_基线灰度": "0_baseline", "1a_局部zscore": "1a_local_zscore",
                   "1b_顶帽": "1b_tophat", "1c_Retinex": "1c_Retinex",
                   "2a_FFT高通": "2a_FFT_highpass", "2b_同态滤波": "2b_homomorphic",
                   "2c_梯度域": "2c_gradient", "3a_LBP": "3a_LBP",
                   "3b_SSIM结构": "3b_SSIM_struct"}
    th = 480
    cells = []
    for name in methods:
        m = res_maps[name]
        m = cv2.resize(m, (int(m.shape[1] * th / m.shape[0]), th))
        bar = np.full((36, m.shape[1]), 30, np.uint8)
        cv2.putText(bar, ascii_label[name], (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cells.append(np.vstack([bar, np.clip(m * 255, 0, 255).astype(np.uint8)]))
    cw = max(c.shape[1] for c in cells)
    cells = [cv2.copyMakeBorder(c, 0, 0, 0, cw - c.shape[1], cv2.BORDER_CONSTANT) for c in cells]
    grid = np.vstack([np.hstack(cells[i:i+3]) for i in range(0, 9, 3)])
    save_img(os.path.join(OUT_DIR, "0_all_methods_grid.png"), grid)

    lines = [f"{'方法':16s} {'RMS':>9s} {'P99':>9s} {'RMS/P99':>9s} {'耗时s':>7s}"]
    for name, rms, p99, ratio, dt in rows:
        lines.append(f"{name:16s} {rms:9.2f} {p99:9.2f} {ratio:9.3f} {dt:7.1f}")
    lines.append("\n注: RMS=残差噪声底(越低光照去除越净); RMS/P99=噪声底/峰值比(越低缺陷对比越好)")
    summary = "\n".join(lines)
    print(summary)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as fp:
        fp.write(summary)
    log("完成")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
TPS 可行性合成验证：
对模板图施加已知低频平滑形变场（模拟蚀刻容差/翘曲）+ 微旋转，生成合成采样图，
对比 纯单应(H) vs 单应+TPS(H+TPS) 两条链路对真值形变场的恢复残差，
输出 改善倍数曲线、门控阈值建议、误差场可视化。

评估口径：对真值前向映射 f(t) = R·t + d(t) 的稠密配准误差（工作分辨率，模板半分辨率）。
"""
import cv2
import numpy as np
import os, time, json, zlib
from scipy.interpolate import RBFInterpolator

TEMPLATE_PATH = r"D:\Users\Administrator\Desktop\weights222\object_region_full_res.jpg"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "synthetic_tps_output")
os.makedirs(OUT_DIR, exist_ok=True)

DOWNSCALE = 0.5            # 工作分辨率 = 模板半分辨率 (~7645x4076)
PATCH = 256                # NCC 匹配 patch 边长
STRIDE = 128               # 匹配网格间距（需 << 形变波长，且保证局部平面拟合偏差 < 阈值）
SEARCH_R = 80              # 每点 2D 搜索半径(px)
NCC_MIN = 0.55             # 峰值 NCC 门槛
OUTLIER_DEV = 10.0         # 局部平面拟合残差剔除阈值(px)
EVAL_STRIDE = 8            # 残差评估网格间距
RBF_NEIGHBORS = 64
LAMBDA_GRID = np.logspace(-4, 3, 10)
GATE_MARGIN = 0.95          # 留出集上 TPS 需优于纯 H 5% 以上才启用
SMALL_PATCH = 96            # 第三遍小 patch（锚点已准，缩小 patch 以适应场内梯度）
SMALL_SEARCH = 32
SMALL_NCC = 0.40

AMPLITUDES_SIN = [0, 10, 20, 40]      # 全局形变幅值(px, 峰值)
AMPLITUDES_LOCAL = [5, 10, 20]        # 局部形变幅值
FIELDS = {"coarse": ("sin", 0.5),     # 全局正弦场, 波长 = frac * 图宽
          "mid": ("sin", 0.25),
          "local": ("local", 0.5),    # 局部高斯窗(σ=0.16W) + 平滑波纹
          "localmid": ("local", 0.25)}  # 局部 + 较短波纹（能力边界探测）
LOCAL_CENTER = (0.68, 0.30)           # bump 中心(归一化 x,y)
LOCAL_SIGMA = 0.16                    # bump 半径(×W)
ROT_DEG = 0.25                    # 全局微旋转，保证 H 有实际可估的全局项
GAIN, BIAS, BLUR_SIGMA, NOISE_SIGMA = 0.7, 10.0, 1.2, 4.0


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------- 真值形变场 ----------------
def make_field(W, H, A, kind, param, seed):
    """生成位移场 d(t) (H,W,2) float32，峰值归一到 A。

    kind="sin":  全局多正弦叠加，波长 = param * W
    kind="local": 高斯窗局部形变（模拟局部蚀刻容差），窗内叠加波纹，波长 = param * W
    """
    rng = np.random.default_rng(seed)
    xn = 2 * np.pi * np.arange(W) / (param * W)
    yn = 2 * np.pi * np.arange(H) / (param * W)
    Xn, Yn = np.meshgrid(xn, yn)
    p = rng.uniform(0, 2 * np.pi, 8)
    ux = 0.7 * np.sin(Xn + p[0]) * np.cos(Yn + p[1]) + 0.3 * np.sin(2 * Xn + p[2]) * np.cos(2 * Yn + p[3])
    uy = 0.7 * np.cos(Xn + p[4]) * np.sin(Yn + p[5]) + 0.3 * np.sin(2 * Xn + p[6]) * np.cos(2 * Yn + p[7])
    d = np.stack([ux, uy], axis=-1)
    if kind == "local":
        X, Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))  # (H,W)
        cx0, cy0 = LOCAL_CENTER[0] * W, LOCAL_CENTER[1] * H
        sig = LOCAL_SIGMA * W
        bump = np.exp(-(((X - cx0) ** 2 + (Y - cy0) ** 2) / (2 * sig * sig)))
        d = d * bump[..., None]
    m = np.max(np.linalg.norm(d, axis=-1))
    if m == 0:
        return np.zeros((H, W, 2), np.float32)
    return (d * (A / m)).astype(np.float32)


def forward_map(W, H, d, rot_deg, scale):
    """真值前向映射 s = R·t + d(t)，返回稠密 sample 坐标场 (H,W,2) float32。"""
    Y, X = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
    th = np.deg2rad(rot_deg)
    Rm = np.array([[scale * np.cos(th), -scale * np.sin(th)],
                   [scale * np.sin(th),  scale * np.cos(th)]])
    cx, cy = W / 2.0, H / 2.0
    sx = cx + Rm[0, 0] * (X - cx) + Rm[0, 1] * (Y - cy) + d[..., 0]
    sy = cy + Rm[1, 0] * (X - cx) + Rm[1, 1] * (Y - cy) + d[..., 1]
    return np.stack([sx, sy], axis=-1).astype(np.float32), Rm


def synthesize_sample(tpl, d, Rm):
    """由 s = R·t + d(t) 的逆映射生成合成采样图（光度学：增益/偏置/模糊/噪声）。

    remap 需要逆向映射 t(s)：迭代 t = c + Rinv·(s - d(t) - c)，定点收敛。
    """
    h, w = tpl.shape
    cx, cy = w / 2.0, h / 2.0
    Rinv = np.linalg.inv(Rm)
    Y, X = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    s = np.stack([X, Y], axis=-1)
    d32 = d.astype(np.float32)
    t = s.copy()
    for _ in range(4):
        tx = t[..., 0].astype(np.float32)
        ty = t[..., 1].astype(np.float32)
        dt = cv2.remap(d32[..., 0], tx, ty, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        dty = cv2.remap(d32[..., 1], tx, ty, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        sd_x = (s[..., 0] - dt - cx).astype(np.float32)
        sd_y = (s[..., 1] - dty - cy).astype(np.float32)
        t = np.stack([cx + Rinv[0, 0] * sd_x + Rinv[0, 1] * sd_y,
                      cy + Rinv[1, 0] * sd_x + Rinv[1, 1] * sd_y], axis=-1).astype(np.float32)
    map_x = t[..., 0].astype(np.float32)
    map_y = t[..., 1].astype(np.float32)
    out = cv2.remap(tpl, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out = cv2.GaussianBlur(out, (0, 0), BLUR_SIGMA)
    out = out * GAIN + BIAS
    rng = np.random.default_rng(7)
    out = np.clip(out + rng.normal(0, NOISE_SIGMA, out.shape), 0, 255).astype(np.uint8)
    return out


# ---------------- NCC 网格匹配 ----------------
def subpixel_peak(score):
    """3x3 抛物面亚像素峰。返回 (dx, dy, peak)。"""
    py, px = np.unravel_index(np.argmax(score), score.shape)
    if py <= 0 or px <= 0 or py >= score.shape[0] - 1 or px >= score.shape[1] - 1:
        return float(px), float(py), float(score[py, px])
    def para(m1, c, p1):
        den = (m1 - 2 * c + p1)
        return 0.0 if den == 0 else 0.5 * (m1 - p1) / den
    dx = para(score[py, px - 1], score[py, px], score[py, px + 1])
    dy = para(score[py - 1, px], score[py, px], score[py + 1, px])
    return px + dx, py + dy, float(score[py, px])


def match_grid(tpl, smp, H0=None, search_r=SEARCH_R, patch=PATCH):
    """模板网格 patch 在采样图中 2D NCC 搜索。

    H0 非空时以 H0 预测位置为锚点（粗到细），窗口半径 search_r，patch 可缩小。
    返回 (pts_t[N,2], disp[N,2], score[N])。
    """
    h, w = tpl.shape
    pad = search_r + 4
    smp_pad = cv2.copyMakeBorder(smp, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    HP, WP = smp_pad.shape
    hp2 = patch // 2
    pts_t, disp, sc = [], [], []
    m = SEARCH_R + hp2
    ys = range(m, h - m, STRIDE)
    xs = range(m, w - m, STRIDE)
    for cy in ys:
        for cx in xs:
            if H0 is None:
                pcx, pcy = float(cx), float(cy)
            else:
                p = H0 @ np.array([cx, cy, 1.0])
                pcx, pcy = p[0] / p[2], p[1] / p[2]
            x0 = int(round(pcx - hp2))
            y0 = int(round(pcy - hp2))
            x0 = min(max(x0, 0), WP - patch - 2 * pad)
            y0 = min(max(y0, 0), HP - patch - 2 * pad)
            patch_img = tpl[cy - hp2: cy + hp2, cx - hp2: cx + hp2]
            win = smp_pad[y0: y0 + patch + 2 * pad, x0: x0 + patch + 2 * pad]
            if patch_img.shape != (patch, patch) or win.shape[:2] != (patch + 2 * pad, patch + 2 * pad):
                continue
            score = cv2.matchTemplate(win, patch_img, cv2.TM_CCOEFF_NORMED)
            mx, my, pk = subpixel_peak(score)
            # 匹配 patch 中心 = 锚点位移 + 窗口内峰偏移
            disp.append((pcx - cx + (mx - pad), pcy - cy + (my - pad)))
            pts_t.append((cx, cy))
            sc.append(pk)
    return np.array(pts_t, np.float64), np.array(disp, np.float64), np.array(sc)


def consistency_filter(pts, disp, score, ncc_min=NCC_MIN):
    """NCC 门槛 + 局部平面拟合一致性剔除。

    用 3x3 网格邻居(不含自身)对每点位移做一阶平面拟合，中心点偏离拟合面
    超过 OUTLIER_DEV 视为误匹配。平面拟合可容纳平滑形变场的大梯度。
    """
    keep = score >= ncc_min
    gx = np.round(pts[:, 0] / STRIDE).astype(int)
    gy = np.round(pts[:, 1] / STRIDE).astype(int)
    gw, gh = gx.max() + 1, gy.max() + 1
    n = len(pts)
    # 网格 -> 点索引
    idx_grid = np.full((gh, gw), -1, dtype=int)
    idx_grid[gy, gx] = np.arange(n)
    keep2 = keep.copy()
    for i in range(n):
        if not keep[i]:
            keep2[i] = False
            continue
        nb = []
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):   # 仅 4 邻域，控制平面拟合距离
            y, x = gy[i] + dy, gx[i] + dx
            if 0 <= y < gh and 0 <= x < gw and idx_grid[y, x] >= 0:
                j = idx_grid[y, x]
                if keep[j]:
                    nb.append(j)
        if len(nb) < 3:
            continue
        j = np.array(nb)
        px = (gx[j] - gx[i]).astype(np.float64)
        py = (gy[j] - gy[i]).astype(np.float64)
        A_ = np.column_stack([np.ones(len(j)), px, py])
        for c in range(2):
            coef, *_ = np.linalg.lstsq(A_, disp[j, c], rcond=None)
            dev = abs(disp[i, c] - coef[0])
            if dev > OUTLIER_DEV:
                keep2[i] = False
                break
    return keep2


# ---------------- H 与 H+TPS ----------------
def fit_homography(pts_t, disp):
    src = pts_t
    dst = pts_t + disp
    Hm, mask = cv2.findHomography(src.reshape(-1, 1, 2).astype(np.float32),
                                  dst.reshape(-1, 1, 2).astype(np.float32),
                                  cv2.USAC_MAGSAC, 6.0, maxIters=5000)
    return Hm, mask.ravel().astype(bool)


def fit_tps(pts_t, disp, Hm):
    """残差 TPS：sample_pos = H(t) + r(t)，λ 留出法选择。

    加权评估：按 |H残差| 加权（只在 H 失效的区域考核），避免局部形变被
    大面积零形变区域的全局 RMSE 稀释。门控：加权 TPS 需优于纯 H 5% 以上。
    """
    src = np.c_[pts_t, np.ones(len(pts_t))] @ Hm.T
    src = src[:, :2] / src[:, 2:3]
    resid = (pts_t + disp) - src
    # 坐标归一化，改善 RBF 数值条件
    scale = np.array([pts_t[:, 0].max(), pts_t[:, 1].max()])
    tn = pts_t / scale
    n = len(tn)
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    n_tr = int(0.8 * n)
    tr, te = idx[:n_tr], idx[n_tr:]
    rw = np.linalg.norm(resid, axis=1)
    w = rw / max(rw.mean(), 1e-9)
    w_te = w[te]
    rmse_h = float(np.sqrt(np.mean(w_te * np.sum(resid[te] ** 2, axis=1))))
    best, best_lam = np.inf, None
    for lam in LAMBDA_GRID:
        rbf = RBFInterpolator(tn[tr], resid[tr], smoothing=lam,
                              kernel="thin_plate_spline", neighbors=RBF_NEIGHBORS)
        pred = rbf(tn[te])
        rmse = float(np.sqrt(np.mean(w_te * np.sum((pred - resid[te]) ** 2, axis=1))))
        if rmse < best:
            best, best_lam = rmse, lam
    if best >= GATE_MARGIN * rmse_h:
        return None, best_lam, best, scale, rmse_h
    rbf = RBFInterpolator(tn, resid, smoothing=best_lam,
                          kernel="thin_plate_spline", neighbors=RBF_NEIGHBORS)
    return rbf, best_lam, best, scale, rmse_h


def eval_errors(W, H, fwd_true, Hm, rbf, rbf_scale):
    """在稠密评估网格上对比 纯H 与 H+TPS 对真值映射的误差。"""
    xs = np.arange(0, W, EVAL_STRIDE, dtype=np.float64)
    ys = np.arange(0, H, EVAL_STRIDE, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    t = np.stack([X.ravel(), Y.ravel()], axis=-1)
    tgt = fwd_true[::EVAL_STRIDE, ::EVAL_STRIDE].reshape(-1, 2)
    th = np.c_[t, np.ones(len(t))] @ Hm.T
    hmap = th[:, :2] / th[:, 2:3]
    err_h = np.linalg.norm(hmap - tgt, axis=1)
    if rbf is not None:
        pred = hmap + rbf(t / rbf_scale)
        err_t = np.linalg.norm(pred - tgt, axis=1)
    else:
        err_t = err_h.copy()
    return err_h, err_t, X.shape


def stats(e):
    return dict(median=float(np.median(e)), p95=float(np.percentile(e, 95)),
                rms=float(np.sqrt(np.mean(e ** 2))), p99=float(np.percentile(e, 99)))


def main():
    log("读取模板…")
    tpl0 = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
    if tpl0 is None:
        raise SystemExit(f"模板读取失败: {TEMPLATE_PATH}")
    Wf = int(tpl0.shape[1] * DOWNSCALE)
    Hf = int(tpl0.shape[0] * DOWNSCALE)
    tpl = cv2.resize(tpl0, (Wf, Hf), interpolation=cv2.INTER_AREA)
    log(f"工作分辨率 {Wf}x{Hf}")

    results = []
    viz_done = False
    local_viz_done = False
    for name, (kind, param) in FIELDS.items():
        amps = AMPLITUDES_SIN if kind == "sin" else AMPLITUDES_LOCAL
        for A in amps:
            tag = f"{name}_A{A}"
            log(f"== {tag} ==")
            d = make_field(Wf, Hf, A, kind, param, seed=zlib.crc32(tag.encode()))
            fwd, Rm = forward_map(Wf, Hf, d, ROT_DEG, 1.0004)
            t0 = time.time()
            smp = synthesize_sample(tpl, d, Rm)
            t_synth = time.time() - t0

            t0 = time.time()
            # 第一遍：全窗口搜索 + 过滤 + MAGSAC H
            pts, disp, sc = match_grid(tpl, smp)
            keep = consistency_filter(pts, disp, sc)
            pts, disp = pts[keep], disp[keep]
            Hm, _ = fit_homography(pts, disp)
            # 第二遍：以 H 预测为锚点小窗重搜（消除周期纹理误锁定），再过滤 + 重拟合 H
            pts2, disp2, sc2 = match_grid(tpl, smp, H0=Hm, search_r=48)
            keep2 = consistency_filter(pts2, disp2, sc2)
            n1 = int(keep.sum())
            pts, disp = pts2[keep2], disp2[keep2]
            Hm, _ = fit_homography(pts, disp)
            # 第三遍：小 patch 精搜（patch 内形变梯度更小，挽回大幅值区域）
            pts3, disp3, sc3 = match_grid(tpl, smp, H0=Hm, search_r=SMALL_SEARCH, patch=SMALL_PATCH)
            keep3 = consistency_filter(pts3, disp3, sc3, ncc_min=SMALL_NCC)
            n2 = len(pts)
            pts, disp = pts3[keep3], disp3[keep3]
            Hm, _ = fit_homography(pts, disp)
            t_match = time.time() - t0
            log(f"匹配点 pass1 {n1} -> pass2 {n2} -> pass3 {len(pts)}, 匹配耗时 {t_match:.1f}s")
            t0 = time.time()
            rbf, lam, hold_rmse, rbf_scale, hold_rmse_h = fit_tps(pts, disp, Hm)
            t_tps = time.time() - t0
            log(f"TPS {'启用' if rbf is not None else '门控禁用'} "
                f"(holdout: TPS {hold_rmse:.2f}px vs H {hold_rmse_h:.2f}px, λ={lam})")

            err_h, err_t, grid_shape = eval_errors(Wf, Hf, fwd, Hm, rbf, rbf_scale)
            sh, st = stats(err_h), stats(err_t)
            gain_rms = sh["rms"] / max(st["rms"], 1e-9)
            gain_p95 = sh["p95"] / max(st["p95"], 1e-9)
            results.append(dict(tag=tag, A=A, field=name, n_match=int(len(pts)),
                                match_s=round(t_match, 2), tps_s=round(t_tps, 2),
                                synth_s=round(t_synth, 2), lam=lam, hold_rmse=hold_rmse,
                                H=sh, TPS=st, gain_rms=gain_rms, gain_p95=gain_p95))
            log(f"纯H  RMS={sh['rms']:.2f} P95={sh['p95']:.2f} | "
                f"H+TPS RMS={st['rms']:.2f} P95={st['p95']:.2f} | "
                f"改善 RMSx{gain_rms:.2f} P95x{gain_p95:.2f} (λ={lam:.1e})")

            if not viz_done and A == 20 and name == "coarse":
                plot_field_viz(tag, Wf, Hf, d, fwd, Hm, rbf, rbf_scale, err_h, err_t, grid_shape)
                viz_done = True
            if not local_viz_done and kind == "local" and A == 10:
                save_local_viz(tag, tpl, smp, pts, disp, Hm, rbf, rbf_scale, Wf, Hf)
                local_viz_done = True

    write_outputs(results)


def plot_field_viz(tag, W, H, d_true, fwd, Hm, rbf, rbf_scale, err_h, err_t, grid_shape):
    """真值场 vs TPS 估计场 + 两种方法误差图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = np.arange(0, W, EVAL_STRIDE, dtype=np.float64)
    ys = np.arange(0, H, EVAL_STRIDE, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    t = np.stack([X.ravel(), Y.ravel()], axis=-1)
    th = np.c_[t, np.ones(len(t))] @ Hm.T
    hmap = th[:, :2] / th[:, 2:3]
    d_true_vec = fwd[::EVAL_STRIDE, ::EVAL_STRIDE].reshape(-1, 2) - hmap  # 去全局后的真值形变
    d_tps_vec = rbf(t / rbf_scale) if rbf is not None else np.zeros_like(t)
    mag_true = np.linalg.norm(d_true_vec, axis=1).reshape(grid_shape)
    mag_tps = np.linalg.norm(d_tps_vec, axis=1).reshape(grid_shape)
    emap_h = err_h.reshape(grid_shape)
    emap_t = err_t.reshape(grid_shape)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    vlim = float(max(mag_true.max(), mag_tps.max(), 1e-6))
    for ax, (m, title) in zip(axes[0], [(mag_true, "GT residual disp |d(t)| (affine removed)"),
                                        (mag_tps, "TPS estimated |d(t)|")]):
        im = ax.imshow(m, origin="upper", cmap="turbo", vmin=0, vmax=vlim,
                       extent=[0, W, H, 0], interpolation="nearest")
        ax.set_title(title); fig.colorbar(im, ax=ax, shrink=0.8)
    vmax = float(np.percentile(emap_h, 99))
    for ax, (title, m) in zip(axes[1], [("err_H (px)", emap_h), ("err_H+TPS (px)", emap_t)]):
        im = ax.imshow(m, origin="upper", cmap="magma", vmin=0, vmax=vmax,
                       extent=[0, W, H, 0], interpolation="nearest")
        ax.set_title(f"{title}  median={np.median(m):.2f} p95={np.percentile(m, 95):.2f}")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"Synthetic deformation {tag} (working res, patch stride {STRIDE})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"field_example_{tag}.png"), dpi=130)
    plt.close(fig)


def save_local_viz(tag, tpl, smp, pts, disp, Hm, rbf, rbf_scale, W, H):
    """局部实验可视化：形变采样图、匹配位移场、H/H+TPS 校正对齐对比。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- 1) 模板 vs 形变采样图（缩小并排） ----
    s = 1400.0 / H
    tpl_s = cv2.resize(tpl, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    smp_s = cv2.resize(smp, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    sep = np.full((tpl_s.shape[0], 6), 255, np.uint8)
    pair = np.hstack([tpl_s, sep, smp_s])
    cv2.imwrite(os.path.join(OUT_DIR, f"{tag}_sample_preview.png"), pair)

    # ---- 2) 匹配位移场（矢量画在采样图缩略图上，颜色=位移幅值） ----
    ov = cv2.cvtColor(smp_s, cv2.COLOR_GRAY2BGR)
    mag = np.linalg.norm(disp, axis=1)
    mmax = max(mag.max(), 1e-6)
    for (x, y), (dx, dy), mg in zip(pts, disp, mag):
        f = mg / mmax
        # 绿(小) -> 红(大)
        color = (0, int(255 * (1 - f)), int(255 * f))
        p1 = (int(x * s), int(y * s))
        p2 = (int((x + dx) * s), int((y + dy) * s))
        cv2.line(ov, p1, p2, color, 1, cv2.LINE_AA)
        cv2.circle(ov, p2, 2, color, -1, cv2.LINE_AA)
    ov = cv2.resize(ov, (ov.shape[1] * 2, ov.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(OUT_DIR, f"{tag}_match_overlay.png"), ov)

    # ---- 3) 校正对比：把采样图反投影回模板坐标系，品红-绿叠加（灰=对齐） ----
    r = 2
    xs = np.arange(0, W, r, dtype=np.float64)
    ys = np.arange(0, H, r, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    t = np.stack([X.ravel(), Y.ravel()], axis=-1)
    th = np.c_[t, np.ones(len(t))] @ Hm.T
    hmap = th[:, :2] / th[:, 2:3]
    tpl_r = tpl[::r, ::r]
    map_x = hmap[:, 0].reshape(len(ys), len(xs)).astype(np.float32)
    map_y = hmap[:, 1].reshape(len(ys), len(xs)).astype(np.float32)
    rect_h = cv2.remap(smp, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if rbf is not None:
        pred = hmap + rbf(t / rbf_scale)
        mx2 = pred[:, 0].reshape(len(ys), len(xs)).astype(np.float32)
        my2 = pred[:, 1].reshape(len(ys), len(xs)).astype(np.float32)
        rect_t = cv2.remap(smp, mx2, my2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    else:
        rect_t = rect_h

    def mg_overlay(a, b):
        """模板=绿通道, 校正图=品红(R+B)；对齐处呈灰白，失配处呈绿/品红双影。"""
        return np.dstack([b, a, b])

    o_h, o_t = mg_overlay(tpl_r, rect_h), mg_overlay(tpl_r, rect_t)
    cx0, cy0 = int(LOCAL_CENTER[0] * W / r), int(LOCAL_CENTER[1] * H / r)
    half = 700
    y0, x0 = max(cy0 - half, 0), max(cx0 - half, 0)
    z_tpl = tpl_r[y0:y0 + 2 * half, x0:x0 + 2 * half]
    z_h = o_h[y0:y0 + 2 * half, x0:x0 + 2 * half]
    z_t = o_t[y0:y0 + 2 * half, x0:x0 + 2 * half]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5),
                             gridspec_kw={"height_ratios": [1.6, 1]})
    axes[0, 0].imshow(cv2.cvtColor(cv2.resize(o_h, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("rectify H-only overlay (full)")
    axes[0, 1].imshow(cv2.cvtColor(cv2.resize(o_t, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("rectify H+TPS overlay (full)")
    axes[0, 2].imshow(cv2.cvtColor(cv2.resize(cv2.absdiff(tpl_r, rect_h), None, fx=0.5, fy=0.5), cv2.COLOR_GRAY2RGB))
    axes[0, 2].set_title("|template - rect H| (full)")
    for ax, (im, title) in zip(axes[1], [(z_tpl, "template zoom"), (z_h, "H-only zoom"),
                                         (z_t, "H+TPS zoom")]):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
    fig.suptitle(f"Local etch-tolerance {tag} (magenta-green overlay, gray = aligned)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{tag}_rectified_compare.png"), dpi=130)
    plt.close(fig)
    cv2.imwrite(os.path.join(OUT_DIR, f"{tag}_absdiff_H.png"),
                cv2.applyColorMap(cv2.normalize(cv2.absdiff(tpl_r, rect_h), None, 0, 255,
                                                cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLORMAP_INFERNO))
    cv2.imwrite(os.path.join(OUT_DIR, f"{tag}_absdiff_TPS.png"),
                cv2.applyColorMap(cv2.normalize(cv2.absdiff(tpl_r, rect_t), None, 0, 255,
                                                cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLORMAP_INFERNO))


def write_outputs(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 曲线：幅值 vs 改善倍数
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for name in FIELDS:
        rs = [r for r in results if r["field"] == name]
        rs.sort(key=lambda r: r["A"])
        a = [r["A"] for r in rs]
        ax[0].plot(a, [r["gain_rms"] for r in rs], "o-", label=f"{name} RMS gain")
        ax[0].plot(a, [r["gain_p95"] for r in rs], "s--", label=f"{name} P95 gain")
        ax[1].plot(a, [r["TPS"]["rms"] for r in rs], "o-", label=f"{name} TPS RMS")
        ax[1].plot(a, [r["H"]["rms"] for r in rs], "x--", label=f"{name} H-only RMS")
    ax[0].axhline(1.0, color="gray", lw=0.8)
    ax[0].set_xlabel("deformation amplitude A (px)"); ax[0].set_ylabel("improvement factor")
    ax[0].set_title("TPS gain vs deformation amplitude"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].set_xlabel("deformation amplitude A (px)"); ax[1].set_ylabel("residual RMS (px)")
    ax[1].set_title("absolute residual"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "gain_curve.png"), dpi=130)
    plt.close(fig)

    lines = ["TPS 合成形变验证 (工作分辨率=模板半分辨率, 单位 px@0.5x)",
             f"配置: patch={PATCH} stride={STRIDE} searchR={SEARCH_R} rot={ROT_DEG}deg "
             f"gain={GAIN} blur={BLUR_SIGMA} noise={NOISE_SIGMA}", ""]
    hdr = f"{'config':<14}{'n':>5}{'lam':>10}{'H_rms':>8}{'H_p95':>8}{'T_rms':>8}{'T_p95':>8}{'gainRMS':>9}{'gainP95':>9}{'t_match':>9}{'t_tps':>7}"
    lines.append(hdr)
    for r in results:
        lines.append(f"{r['tag']:<14}{r['n_match']:>5}{r['lam']:>10.1e}"
                     f"{r['H']['rms']:>8.2f}{r['H']['p95']:>8.2f}{r['TPS']['rms']:>8.2f}{r['TPS']['p95']:>8.2f}"
                     f"{r['gain_rms']:>9.2f}{r['gain_p95']:>9.2f}{r['match_s']:>9.1f}{r['tps_s']:>7.2f}")
    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()

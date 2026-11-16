# -*- coding: utf-8 -*-
"""
Halcon 级形状匹配实现：
  1) 金字塔式角度搜索：粗网格(2°) -> 细网格(0.25°) -> 抛物线亚像素角度
  2) 平移亚像素：相关面峰值抛物线内插
  3) 方向残差最小二乘精化：全分辨率边缘点，数值雅可比 Gauss-Newton，多峰启动
与简化版(FFT网格搜索)、NCC+MAGSAC 做同口径对比。
"""
import cv2
import numpy as np
import os, time, sys

TPL = r"D:\Users\Administrator\Desktop\weights222\object_region_full_res.jpg"
SMP = r"D:\Users\Administrator\Desktop\weights222\corresponding_sample_region_full_res.jpg"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shape_match_test_output")

F = 2                      # 搜索分辨率 1/F
HP_SIGMA = 8               # 半分辨率高通
EDGE_PCT = 97
REFINE_PCT = 97
N_POINTS = 30000


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def hp(g, s):
    k = int(s * 3) | 1
    return g.astype(np.float32) - cv2.GaussianBlur(g, (k, k), s)


def nms_mask(mag):
    """8邻域非最大抑制"""
    m = mag
    keep = (m >= np.roll(m, 1, 0)) & (m >= np.roll(m, -1, 0)) & \
           (m >= np.roll(m, 1, 1)) & (m >= np.roll(m, -1, 1)) & \
           (m >= np.roll(np.roll(m, 1, 0), 1, 1)) & (m >= np.roll(np.roll(m, 1, 0), -1, 1)) & \
           (m >= np.roll(np.roll(m, -1, 0), 1, 1)) & (m >= np.roll(np.roll(m, -1, 0), -1, 1))
    return keep & (m > 0)


def edge_points(img, pct, max_n=None):
    """亚像素级边缘点（这里取整像素+NMS，方向来自梯度）"""
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, 3)
    mag = np.sqrt(gx * gx + gy * gy)
    keep = nms_mask(mag) & (mag >= np.percentile(mag, pct))
    ys, xs = np.where(keep)
    if max_n and len(ys) > max_n:
        idx = np.random.default_rng(0).choice(len(ys), max_n, replace=False)
        ys, xs = ys[idx], xs[idx]
    theta = np.arctan2(gy[ys, xs], gx[ys, xs])
    w = mag[ys, xs]
    return xs.astype(np.float64), ys.astype(np.float64), theta, w / (w.max() + 1e-9)


def orient_field_fft(img, pct):
    """倍角方向场（ignore_local_polarity 等价），用于 FFT 平移搜索"""
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, 3)
    mag = np.sqrt(gx * gx + gy * gy)
    m = mag >= np.percentile(mag, pct)
    G = np.zeros(img.shape, np.complex128)
    G[m] = ((gx[m] ** 2 - gy[m] ** 2) + 2j * gx[m] * gy[m]) / (mag[m] ** 2)
    return G, int(m.sum())


def fft_search(tpl_hp, smp_hp, angles):
    Gs, _ = orient_field_fft(smp_hp, EDGE_PCT)
    Fs = np.fft.fft2(Gs); Fs[0, 0] = 0
    H, W = smp_hp.shape
    res = []
    for a in angles:
        t_rot = cv2.warpAffine(tpl_hp, cv2.getRotationMatrix2D((W / 2, H / 2), -a, 1.0), (W, H),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        Gt, nt = orient_field_fft(t_rot, EDGE_PCT)
        Ft = np.fft.fft2(Gt); Ft[0, 0] = 0
        C = np.fft.ifft2(Ft * np.conj(Fs)).real / nt
        py, px = np.unravel_index(np.argmax(C), C.shape)
        res.append((float(a), int(px), int(py), float(C[py, px]), C))
    return res


def parabolic(sm, s0, sp, step):
    d = 0.5 * (sm - sp) / (sm - 2 * s0 + sp + 1e-12)
    return np.clip(d, -1, 1) * step


def refine_lsq(P, theta_t, w_t, ang_map, mag_map, a_deg, tx, ty, center, iters=12):
    """方向残差最小二乘：数值雅可比 Gauss-Newton，最小化 Σ w(1-cos(θt+a-θs))"""
    H, W = ang_map.shape
    c = center
    N = len(P)

    def sample_q(a_deg_, tx_, ty_):
        a_ = np.radians(a_deg_)
        R = np.array([[np.cos(a_), -np.sin(a_)], [np.sin(a_), np.cos(a_)]])
        q = (P - c) @ R.T + c + np.array([tx_, ty_])
        qx = np.clip(q[:, 0], 0, W - 1).astype(np.float32)
        qy = np.clip(q[:, 1], 0, H - 1).astype(np.float32)
        th_s = cv2.remap(ang_map, qx, qy, cv2.INTER_LINEAR).reshape(-1)
        m_s = cv2.remap(mag_map, qx, qy, cv2.INTER_LINEAR).reshape(-1)
        return th_s, m_s

    def score(a_deg_, tx_, ty_):
        th_s, m_s = sample_q(a_deg_, tx_, ty_)
        v = m_s > max(1e-6, float(np.percentile(m_s, 60)))
        w = w_t * v
        return float(np.sum(w * np.cos(2 * (theta_t + np.radians(a_deg_) - th_s)))), w, th_s

    s_cur, w_cur, _ = score(a_deg, tx, ty)
    for it in range(iters):
        da, dt = 0.05, 0.75  # 数值微分步长: 0.05°, 0.75px
        # 残差 r_i = w*cos(...) ; 数值雅可比 (N,3)
        def residuals(a_, tx_, ty_):
            th_s, m_s = sample_q(a_, tx_, ty_)
            v = m_s > max(1e-6, float(np.percentile(m_s, 60)))
            return (w_t * v) * np.cos(2 * (theta_t + np.radians(a_) - th_s))
        r0 = residuals(a_deg, tx, ty)
        e1 = residuals(a_deg + da, tx, ty) - residuals(a_deg - da, tx, ty)
        e2 = residuals(a_deg, tx + dt, ty) - residuals(a_deg, tx - dt, ty)
        e3 = residuals(a_deg, tx, ty + dt) - residuals(a_deg, tx, ty - dt)
        for nm, e in (("+a", e1), ("+x", e2), ("+y", e3)):
            if e.ndim != 1 or e.shape[0] != N:
                print(f"  [BAD jacobian col {nm}] shape={e.shape} N={N} "
                      f"th_s_check={sample_q(a_deg, tx, ty)[0].shape}")
        J = np.column_stack([e1, e2, e3]) / (2 * 1.0)
        Hm = J.T @ J + 1e-9 * np.eye(3)
        g = J.T @ (r0 - np.ones(N) * 0)  # 目标: 最大化 S -> 等价残差趋向 w
        # 对 max Σ r 求解: 线化 r ≈ r0 + J δ, 令 r -> w_t (理想满分)
        target = w_t
        delta = np.linalg.solve(Hm, J.T @ (target - r0))
        # 信赖域: 单步限制 (防GN在周期性结构上大步漂移)
        delta[0] = np.clip(delta[0], -0.15, 0.15)
        delta[1:] = np.clip(delta[1:], -3.0, 3.0)
        # 步长阻尼 + 回退
        ok = False
        for k in range(4):
            a_n = a_deg + delta[0]; tx_n = tx + delta[1]; ty_n = ty + delta[2]
            s_n, _, _ = score(a_n, tx_n, ty_n)
            if s_n >= s_cur:
                a_deg, tx, ty, s_cur = a_n, tx_n, ty_n, s_n
                ok = True
                break
            delta *= 0.5
        if not ok or np.linalg.norm(delta[[1, 2]]) < 0.05 and abs(delta[0]) < 0.005:
            break
    return a_deg, tx, ty, s_cur


def refine_progressive(P, theta_t, w_t, sf_hp, center, starts, sigmas=(20.0, 12.0, 6.0, 2.5, 0.0)):
    """Halcon 式金字塔: 每级 = 局部平移窗口穷搜 + 角度局部搜索 + GN 亚像素。
    粗级模糊大->边缘肥厚->捕获域宽; 逐级细化->亚像素精度。"""
    H, W = sf_hp.shape
    N = len(P)
    rng_cache = {}

    def maps(s_b):
        if s_b not in rng_cache:
            sb = cv2.GaussianBlur(sf_hp, (0, 0), s_b) if s_b > 0 else sf_hp
            gxb = cv2.Sobel(sb, cv2.CV_32F, 1, 0, 3)
            gyb = cv2.Sobel(sb, cv2.CV_32F, 0, 1, 3)
            rng_cache[s_b] = (cv2.phase(gxb, gyb).astype(np.float32),
                              cv2.magnitude(gxb, gyb).astype(np.float32))
        return rng_cache[s_b]

    def S_at(am, mm, a_deg_, tx_, ty_):
        a_ = np.radians(a_deg_)
        R = np.array([[np.cos(a_), -np.sin(a_)], [np.sin(a_), np.cos(a_)]])
        q = (P - center) @ R.T + center + np.array([tx_, ty_])
        qx = np.clip(q[:, 0], 0, W - 1).astype(np.float32)
        qy = np.clip(q[:, 1], 0, H - 1).astype(np.float32)
        th_s = cv2.remap(am, qx, qy, cv2.INTER_LINEAR).reshape(-1)
        m_s = cv2.remap(mm, qx, qy, cv2.INTER_LINEAR).reshape(-1)
        v = m_s > max(1e-6, float(np.percentile(m_s, 60)))
        return float(np.sum((w_t * v) * np.cos(2 * (theta_t + a_ - th_s))))

    def grid_search(am, mm, a_deg_, tx_, ty_, win, step):
        cand = [(ox, oy) for ox in range(-win, win + 1, step)
                for oy in range(-win, win + 1, step)]
        scores = [S_at(am, mm, a_deg_, tx_ + ox, ty_ + oy) for ox, oy in cand]
        i = int(np.argmax(scores))
        (bx, by), s0 = cand[i], scores[i]
        # 邻点抛物线亚像素
        def sc(ox, oy):
            return S_at(am, mm, a_deg_, tx_ + ox, ty_ + oy)
        dxs = dys = 0.0
        if win > 0:
            sxm, sxp = sc(bx - step, by), sc(bx + step, by)
            sym, syp = sc(bx, by - step), sc(bx, by + step)
            dxs = parabolic(sxm, s0, sxp, step)
            dys = parabolic(sym, s0, syp, step)
        return tx_ + bx + dxs, ty_ + by + dys, s0

    def angle_search(am, mm, a_deg_, tx_, ty_, win=0.3, step=0.1):
        angs = np.arange(a_deg_ - win, a_deg_ + win + 1e-9, step)
        scores = [S_at(am, mm, float(a_), tx_, ty_) for a_ in angs]
        i = int(np.argmax(scores))
        return float(angs[i]), scores[i]

    best = None
    for (a0_, tx0, ty0) in starts:
        a_, tx_, ty_ = a0_, tx0, ty0
        for li, s_b in enumerate(sigmas):
            am, mm = maps(s_b)
            win, step = (25, 2) if li == 0 else (4, 1)
            for _rep in range(2):  # 平移/角度交替两轮
                tx_, ty_, _ = grid_search(am, mm, a_, tx_, ty_, win, step)
                a_, _ = angle_search(am, mm, a_, tx_, ty_, win=0.4, step=0.05)
            a_, tx_, ty_, sc = refine_lsq(P, theta_t, w_t, am, mm, a_, tx_, ty_, center, iters=6)
            log(f"    级{s_b:.0f}: a={a_:+.3f}° T=({tx_:.1f},{ty_:.1f}) S={sc:.0f}")
        if best is None or sc > best[3]:
            best = (a_, tx_, ty_, sc)
    return best


def main():
    t_start = time.time()
    refine_only = len(sys.argv) > 1 and sys.argv[1] == "--refine-only"
    log("加载 ...")
    tpl = cv2.imread(TPL, 0); smp = cv2.imread(SMP, 0)
    FH, FW = tpl.shape
    tg = cv2.resize(tpl, (FW // F, FH // F), interpolation=cv2.INTER_AREA).astype(np.float32)
    sg = cv2.resize(smp, (FW // F, FH // F), interpolation=cv2.INTER_AREA).astype(np.float32)
    tg_hp, sg_hp = hp(tg, HP_SIGMA), hp(sg, HP_SIGMA)
    Hh, Wh = sg.shape

    # --- 1) 粗搜索 2° ---
    if refine_only:
        a1, dx1, dy1, a_sub, dx_sub, dy_sub = 0.0, -2, -5, 0.049, -2.1, -4.57
        t_coarse = t_fine = 0.0
        log(f"[refine-only] 跳过FFT, init: angle={a_sub} d=({dx_sub},{dy_sub})")
    else:
        # --- 1) 粗搜索 2° ---
        log("粗搜索 2° ...")
        t0 = time.time()
        coarse = fft_search(tg_hp, sg_hp, np.arange(-12, 12.1, 2.0))
        coarse.sort(key=lambda r: -r[3])
        a0 = coarse[0][0]
        t_coarse = time.time() - t0
        log(f"  最优粗角 {a0:+.1f}° ({t_coarse:.1f}s)")

        # --- 2) 细搜索 0.25° ---
        log("细搜索 0.25° ...")
        t0 = time.time()
        fine = fft_search(tg_hp, sg_hp, np.arange(a0 - 1.5, a0 + 1.51, 0.25))
        fine.sort(key=lambda r: -r[3])
        a1, dx1, dy1, s1, C_best = fine[0]
        t_fine = time.time() - t0
        log(f"  细角 {a1:+.2f}° d=({dx1},{dy1}) score={s1:.4f} ({t_fine:.1f}s)")

        # --- 3) 三方向亚像素 ---
        da = 0.25
        s_map = {round(r[0], 2): r[3] for r in fine}
        sm = s_map.get(round(a1 - da, 2), s1); sp = s_map.get(round(a1 + da, 2), s1)
        a_sub = a1 + parabolic(sm, s1, sp, da)
        py, px = dy1, dx1
        cy0, cx0 = C_best[py, px], C_best[py, px]
        sy1 = C_best[max(py - 1, 0), px]; sy2 = C_best[min(py + 1, Hh - 1), px]
        sx1 = C_best[py, max(px - 1, 0)]; sx2 = C_best[py, min(px + 1, Wh - 1)]
        dy_sub = py + parabolic(sy1, cy0, sy2, 1.0)
        dx_sub = px + parabolic(sx1, cx0, sx2, 1.0)
        log(f"  亚像素: angle={a_sub:+.3f}° d=({dx_sub:.2f},{dy_sub:.2f})")

    # --- 4) 最小二乘精化（全分辨率, 多峰启动） ---
    log("最小二乘精化 ...")
    t0 = time.time()
    tf = hp(tpl, HP_SIGMA * F)
    sf = hp(smp, HP_SIGMA * F)
    ex, ey, eth, ew = edge_points(tf, REFINE_PCT, N_POINTS)
    P2 = np.column_stack([ex, ey])
    gx = cv2.Sobel(sf, cv2.CV_32F, 1, 0, 3); gy = cv2.Sobel(sf, cv2.CV_32F, 0, 1, 3)
    ang_map = cv2.phase(gx, gy).astype(np.float32)
    mag_map = cv2.magnitude(gx, gy).astype(np.float32)
    center = np.array([FW / 2.0, FH / 2.0])

    if refine_only:
        starts = [(a_sub, dx_sub * F, dy_sub * F)]
    else:
        # 多峰启动: FFT峰 + 对角偏移 ±12px(扩大捕获域) + 次峰
        Cc = C_best.copy()
        peaks = []
        for _ in range(2):
            p = np.unravel_index(np.argmax(Cc), Cc.shape)
            peaks.append(p)
            Cc[max(p[0] - 32, 0):p[0] + 32, max(p[1] - 32, 0):p[1] + 32] = -1
        py0, px0 = peaks[0]
        dy0 = py0 - (Hh if py0 > Hh / 2 else 0)
        dx0 = px0 - (Wh if px0 > Wh / 2 else 0)
        starts = [(a_sub, dx0 * F + ox, dy0 * F + oy)
                  for ox in (-12, 0, 12) for oy in (-12, 0, 12)]
        pyk, pxk = peaks[1]
        starts.append((a_sub,
                       (pxk - (Wh if pxk > Wh / 2 else 0)) * F,
                       (pyk - (Hh if pyk > Hh / 2 else 0)) * F))
    best = refine_progressive(P2, eth, ew, sf, center, starts)
    a_fin, tx_fin, ty_fin, s_fin = best
    t_ref = time.time() - t0
    log(f"  精化: angle={a_fin:+.3f}° T=({tx_fin:.2f},{ty_fin:.2f}) score={s_fin:.1f} ({t_ref:.1f}s)")

    # --- NCC 基准 ---
    log("NCC 基准 ...")
    t0 = time.time()
    pts, us = [], []
    half = 80
    for ty in range(half + 60, Hh - half - 60, 96):
        for tx in range(half + 60, Wh - half - 60, 96):
            r = cv2.matchTemplate(sg[ty - 60 - half:ty + 60 + half, tx - 60 - half:tx + 60 + half],
                                  tg[ty - half:ty + half, tx - half:tx + half], cv2.TM_CCOEFF_NORMED)
            _, s, _, loc = cv2.minMaxLoc(r)
            if s > 0.6:
                pts.append((tx, ty)); us.append((loc[0] - 60, loc[1] - 60))
    Pn = np.array(pts, float); Un = np.array(us, float)
    from scipy.spatial import cKDTree
    H2, _ = cv2.findHomography(Pn.astype(np.float32), (Pn + Un).astype(np.float32), cv2.USAC_MAGSAC, 3.0)
    for _ in range(3):
        _, nb = cKDTree(Pn).query(Pn, k=12)
        dev = np.linalg.norm(Un - np.median(Un[nb], axis=1), axis=1)
        g = dev < max(6, 4 * (np.median(dev) + 1e-6))
        if g.all(): break
        Pn, Un = Pn[g], Un[g]
        H2, _ = cv2.findHomography(Pn.astype(np.float32), (Pn + Un).astype(np.float32), cv2.USAC_MAGSAC, 3.0)
    t_ncc = time.time() - t0
    S4 = np.diag([F, F, 1]).astype(np.float64)
    H_full = S4 @ H2.astype(np.float64) @ np.diag([1 / F, 1 / F, 1])  # 全分辨率(用于误差指标)
    # 画图用: 半->半 就是 H2 本身; S·H2·S⁻¹ 是全分辨率单应, 平移×2, 不能用于画布
    H_half = H2.astype(np.float64)
    log(f"  {len(Pn)} 点 ({t_ncc:.1f}s)")

    # --- 评估 ---
    rng = np.random.default_rng(7)
    sel = rng.choice(len(Pn), min(1500, len(Pn)), replace=False)
    Pt = Pn[sel] * F; Ps = (Pn + Un)[sel] * F  # 全分辨率

    def errs_fwd(a_deg_, T_):
        a_ = np.radians(a_deg_)
        R = np.array([[np.cos(a_), -np.sin(a_)], [np.sin(a_), np.cos(a_)]])
        q = (Pt - center) @ R.T + center + T_
        return np.linalg.norm(q - Ps, axis=1)

    e_init = errs_fwd(a_sub, np.array([dx_sub * F, dy_sub * F]))
    e_fin = errs_fwd(a_fin, np.array([tx_fin, ty_fin]))
    e_h = np.linalg.norm(cv2.perspectiveTransform(Pt.reshape(-1, 1, 2).astype(np.float32),
                                                  H_full).reshape(-1, 2) - Ps, axis=1)
    ang_h = np.degrees(np.arctan2(H_full[1, 0] - H_full[0, 1], H_full[0, 0] + H_full[1, 1]))

    def stat(e):
        return f"median={np.median(e):.2f} P95={np.percentile(e,95):.2f} RMS={np.sqrt((e**2).mean()):.2f}"

    # 诊断: NCC位姿处的方向得分 vs 精化位姿 —— 区分"优化器被困"还是"得分函数有偏"
    aHr = np.radians(ang_h)
    R_H = np.array([[np.cos(aHr), -np.sin(aHr)], [np.sin(aHr), np.cos(aHr)]])
    T_eq = H_full[:2, 2] - center + R_H @ center
    qh = (P2 - center) @ R_H.T + center + T_eq
    qhx = np.clip(qh[:, 0], 0, FW - 1).astype(np.float32)
    qhy = np.clip(qh[:, 1], 0, FH - 1).astype(np.float32)
    thH = cv2.remap(ang_map, qhx, qhy, cv2.INTER_LINEAR).reshape(-1)
    mH = cv2.remap(mag_map, qhx, qhy, cv2.INTER_LINEAR).reshape(-1)
    vH = mH > max(1e-6, float(np.percentile(mH, 60)))
    S_H = float(np.sum((ew * vH) * np.cos(2 * (eth + aHr - thH))))
    log(f"  诊断: S(NCC位姿)={S_H:.0f} vs S(LSQ精化)={s_fin:.0f}  "
        f"({'得分函数有偏,方法受限' if S_H <= s_fin else '优化器困在局部最优'})")
    log("全分辨率逐点误差(px):")
    log(f"  亚像素网格版 : {stat(e_init)}")
    log(f"  最小二乘精化 : {stat(e_fin)}")
    log(f"  NCC+MAGSAC   : {stat(e_h)}")
    log(f"角度: LSQ={a_fin:+.3f}° vs H={ang_h:+.3f}°; 耗时 形状匹配={t_coarse+t_fine+t_ref:.1f}s NCC={t_ncc:.1f}s")

    # --- 可视化: LSQ位姿(青) vs NCC(品红) ---
    a_ = np.radians(a_fin)
    R = np.array([[np.cos(a_), -np.sin(a_)], [np.sin(a_), np.cos(a_)]])
    c_half = np.array([FW / (2.0 * F), FH / (2.0 * F)])  # 模板中心(半分辨率)
    T_half = np.array([tx_fin, ty_fin]) / F
    M_fin = np.eye(2, 3)
    M_fin[:, :2] = R                       # 线性部分=旋转本身, 不能再除F
    M_fin[:, 2] = (np.eye(2) - R) @ c_half + T_half
    edges = cv2.Canny(tg.astype(np.uint8), 50, 150)
    col = np.zeros((Hh, Wh, 3), np.uint8); col[..., 1] = edges
    w_lsq = cv2.warpAffine(col, M_fin, (Wh, Hh))[..., 1] > 0
    w_ncc = cv2.warpPerspective(col, H_half, (Wh, Hh))[..., 1] > 0
    base = cv2.cvtColor(sg, cv2.COLOR_GRAY2BGR)
    ov = base.copy(); ov[w_lsq] = (255, 255, 0); ov[w_ncc] = (255, 0, 255)
    ok, buf = cv2.imencode(".png", ov); buf.tofile(os.path.join(OUT, "halcon_vs_ncc_full.png"))
    zy, zx = 640, 520
    zoom = ov[zy:zy + 1000, zx:zx + 1500]
    ok, buf = cv2.imencode(".png", cv2.resize(zoom, (1800, 1200)))
    buf.tofile(os.path.join(OUT, "halcon_vs_ncc_zoom.png"))

    lines = [
        f"coarse {t_coarse:.1f}s + fine {t_fine:.1f}s + LSQ {t_ref:.1f}s = {t_coarse+t_fine+t_ref:.1f}s | NCC {t_ncc:.1f}s",
        f"角度: grid={a_sub:+.3f} LSQ={a_fin:+.3f} H={ang_h:+.3f}",
        f"err(px) subpix-grid: {stat(e_init)}",
        f"err(px) LSQ refined: {stat(e_fin)}",
        f"err(px) NCC+MAGSAC : {stat(e_h)}",
    ]
    print("\n".join(lines))
    with open(os.path.join(OUT, "halcon_grade_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log("完成")


if __name__ == "__main__":
    main()

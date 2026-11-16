# -*- coding: utf-8 -*-
"""
补测重跑：Fourier-Mellin 相位相关（标准 FFT 域实现）。
与上轮差异：
  1) log-polar 改在幅度谱上做（平移不变），而非空间域 warpPolar（上轮的根本缺陷）；
  2) 内置合成自标定：已知 (θ,s,t) 变换跑一遍管线，自动解出角度/缩放/平移的符号与量纲因子，
     再应用到真实对——不再依赖对 OpenCV 约定的手工推导；
  3) 顺带测 FM 初值 → ECC 金字塔 的完整全局方案。
输出：retest_phase_corr_output/
"""
import cv2
import numpy as np
import os, time, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import synthetic_tps_validation as stv

OUT_DIR = os.path.join(HERE, "retest_phase_corr_output")
os.makedirs(OUT_DIR, exist_ok=True)
DOWNSCALE = 0.5
POLAR = 720          # log-polar 尺寸（角度 × 半径同值）
HP_SIGMA = 8.0       # 幅度谱高通 sigma


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


def compose_H(theta_deg, scale, tx, ty, W, H):
    """sample = template 绕图心旋转 θ、缩放 s、平移 (tx,ty) 后的像；H 映射 template→sample。"""
    th = np.deg2rad(theta_deg)
    R = scale * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    c = np.array([W / 2.0, H / 2.0])
    Hm = np.eye(3)
    Hm[:2, :2] = R
    Hm[:2, 2] = c + np.array([tx, ty]) - R @ c
    return Hm


def _hp_hann(g, win, sigma):
    g = g.astype(np.float64)
    k = int(sigma * 3) | 1
    return (g - cv2.GaussianBlur(g, (k, k), sigma)) * win


def fm_measure(a, b, polar=POLAR):
    """Fourier-Mellin 原始测量（约定未定标）。
    轴序实测（debug_fm.py）：warpPolar 输出 y 轴=角度、x 轴=对数半径。
    非方形图 DFT 频率索引为椭圆（旋转在索引空间不是圆），故 FFT 前裁中心正方形。
    返回 (th_raw_deg, ls_raw, resp_polar)。"""
    h, w = a.shape
    s = min(w, h)
    y0, x0 = (h - s) // 2, (w - s) // 2
    asq = np.ascontiguousarray(a[y0:y0 + s, x0:x0 + s], dtype=np.float32)
    bsq = np.ascontiguousarray(b[y0:y0 + s, x0:x0 + s], dtype=np.float32)
    win = cv2.createHanningWindow((s, s), cv2.CV_32F)
    Fa = cv2.dft(asq * win, flags=cv2.DFT_COMPLEX_OUTPUT)
    Fb = cv2.dft(bsq * win, flags=cv2.DFT_COMPLEX_OUTPUT)
    m1 = np.fft.fftshift(np.log(cv2.magnitude(Fa[..., 0], Fa[..., 1]).astype(np.float64) + 1e-6))
    m2 = np.fft.fftshift(np.log(cv2.magnitude(Fb[..., 0], Fb[..., 1]).astype(np.float64) + 1e-6))
    # 预高通（去掉 DC 强块）+ 限制动态范围
    k = int(HP_SIGMA * 3) | 1
    m1 = m1 - cv2.GaussianBlur(m1, (k, k), HP_SIGMA)
    m2 = m2 - cv2.GaussianBlur(m2, (k, k), HP_SIGMA)
    m1 = np.clip(m1, None, 5.0).astype(np.float32)
    m2 = np.clip(m2, None, 5.0).astype(np.float32)
    center = (s // 2, s // 2)
    max_r = s // 2
    p1 = cv2.warpPolar(m1, (polar, polar), center, max_r, cv2.WARP_POLAR_LOG)
    p2 = cv2.warpPolar(m2, (polar, polar), center, max_r, cv2.WARP_POLAR_LOG)
    # 注意：不做径向带限——裁掉内侧列会偏置相位相关的亚像素估计（实测 ls 偏 18%）
    pwin = cv2.createHanningWindow((p1.shape[1], p1.shape[0]), cv2.CV_64F)
    pa = p1.astype(np.float64) * pwin
    pb = p2.astype(np.float64) * pwin
    (dx, dy), resp = cv2.phaseCorrelate(pa, pb)
    th_raw = -dy * 360.0 / polar      # 角度：y 轴，5°=10px
    ls_raw = -dx * np.log(max_r) / polar  # 对数缩放：x 轴
    return th_raw, ls_raw, resp, max_r, polar


def fm_refine_translation(a, b, theta_deg, scale):
    """撤销 (θ,s) 后测平移。
    关键实证（debug_fm5.py）：撤销必须用显式 inv(compose_H(θ,s,0,0)) 作 warpAffine 的 M，
    getRotationMatrix2D 构造的矩阵与 H⁻¹ 差一个常数项，几何对不上。
    bc 内容相对 a 的位移 = −A⁻¹t → t = −A·(dx,dy)。返回 (tx, ty, resp)。"""
    h, w = a.shape
    H_rs = compose_H(theta_deg, scale, 0.0, 0.0, w, h)
    Hinv = np.linalg.inv(H_rs)
    M = np.ascontiguousarray(Hinv[:2] / Hinv[2, 2], dtype=np.float64)
    bc = cv2.warpAffine(b, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    win = cv2.createHanningWindow((w, h), cv2.CV_64F)
    (dx, dy), resp = cv2.phaseCorrelate(_hp_hann(a, win, 24.0), _hp_hann(bc, win, 24.0))
    th = np.deg2rad(theta_deg)
    A = scale * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    t = -A @ np.array([dx, dy])
    return float(t[0]), float(t[1]), resp


def calibrate(a, W, H, name):
    """合成已知变换自标定。返回 (ok, resid, k_th, k_sl, k_tx, k_ty)。"""
    th_t, s_t, t_t = 3.0, 1.04, (20.0, -13.0)
    H_true = compose_H(th_t, s_t, t_t[0], t_t[1], W, H)
    b = cv2.warpPerspective(a, H_true, (W, H), flags=cv2.INTER_LINEAR)
    th_meas_raw, ls_meas_raw, resp, _, _ = fm_measure(a, b)
    log(f"[{name}] 自标定 raw: dθ={th_meas_raw:+.3f}° (真{th_t:+.1f}), log s={ls_meas_raw:+.4f} "
        f"(真{np.log(s_t):+.4f}), resp={resp:.3f}")
    # 符号/量纲只在 ±{0.5,1,2} 里找；找不到说明极相关找错峰，直接判失败
    k_th = th_t / th_meas_raw if abs(th_meas_raw) > 1e-6 else 0.0
    k_sl = np.log(s_t) / ls_meas_raw if abs(ls_meas_raw) > 1e-6 else 0.0
    ok = k_th != 0 and k_sl != 0
    if ok:
        for cand in (1.0, -1.0, 0.5, -0.5, 2.0, -2.0):
            if abs(k_th - cand) < 0.05 * abs(cand) or abs(1 / k_th - cand) < 0.05 * abs(cand):
                k_th = cand if abs(k_th - cand) < 0.05 * abs(cand) else 1.0 / cand
                break
        else:
            ok = False
    if ok:
        for cand in (1.0, -1.0):
            if abs(k_sl - cand) < 0.1:
                k_sl = cand
                break
        else:
            ok = False
    if not ok:
        log(f"[{name}] 自标定失败（比值 k_th={k_th:.3f} k_sl={k_sl:.3f} 非符号/量纲因子 → 极相关找错峰）")
        return False, None, 0, 0, 0, 0

    # 平移步定标
    th_fix = th_meas_raw * k_th
    s_fix = float(np.exp(ls_meas_raw * k_sl))
    best, k_best = None, (1.0, 1.0)
    for kx in (1.0, -1.0):
        for ky in (1.0, -1.0):
            dx2, dy2, _ = fm_refine_translation(a, b, th_fix, s_fix)
            H_try = compose_H(th_fix, s_fix, dx2 * kx, dy2 * ky, W, H)
            dd = dense_dev(H_try, H_true, W, H, stride=16)["median"]
            if best is None or dd < best:
                best, k_best = dd, (kx, ky)
    ok = best < 5.0
    log(f"[{name}] 自标定: k_th={k_th:+.0f} k_sl={k_sl:+.0f} k_t={k_best}, 残差 med={best:.2f}px → "
        f"{'通过' if ok else '未通过(<5px)'}")
    return ok, best, k_th, k_sl, k_best[0], k_best[1]


def _zncc_verify(a, b, Hm, scale_down=0.25):
    """按 H 把 b 翘到 a 坐标系，分块局部 ZNCC 取中位数——候选 H 打分用。
    全局 ZNCC 不可用：模板内容大量重复 + 真实局部形变（|disp| med 8.9px），
    错位 726px 的候选靠"对齐大量重复结构"反而全局相关更高（实测 0.41 vs 真值 0.32）。
    分块中位数只认"大多数块各自对齐"，重复结构骗不过。"""
    h, w = a.shape
    warp = cv2.warpPerspective(b, np.linalg.inv(Hm), (w, h),
                               flags=cv2.INTER_LINEAR, borderValue=0)
    s = scale_down
    aw = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    bw = cv2.resize(warp, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ch, cw = aw.shape
    blk, srch, step = 32, 8, 64
    scores = []
    for by in range(srch, ch - blk - srch, step):
        for bx in range(srch, cw - blk - srch, step):
            patch = aw[by:by + blk, bx:bx + blk]
            if patch.std() < 4.0:
                continue
            win = bw[by - srch:by + blk + srch, bx - srch:bx + blk + srch]
            if win.std() < 4.0 or win.shape[0] < blk or win.shape[1] < blk:
                continue
            r = cv2.matchTemplate(win, patch, cv2.TM_CCOEFF_NORMED)
            scores.append(float(r.max()))
    if len(scores) < 20:
        return -1.0
    return float(np.median(scores))


def fm_full(a, b, cal, W, H):
    """定标后的完整 FM：返回 (H, dθ_deg, scale, t, 响应)。
    平移通道取两个候选（相位相关峰 / 先验 t=0），ZNCC 验证择优——真实对上
    相位相关平移峰会锁到重复纹理谐波（实测 resp=0.03、t=726px）。"""
    _, _, k_th, k_sl, k_tx, k_ty = cal
    th_raw, ls_raw, resp, _, _ = fm_measure(a, b)
    th = th_raw * k_th
    ls = ls_raw * k_sl
    s = float(np.exp(ls))
    tx, ty, resp_t = fm_refine_translation(a, b, th, s)
    cands = [(tx * k_tx, ty * k_ty, resp_t, "pc")]
    cands.append((0.0, 0.0, 0.0, "prior0"))
    best, best_score = None, -2.0
    for tx_, ty_, r_, src in cands:
        Hc = compose_H(th, s, tx_, ty_, W, H)
        sc = _zncc_verify(a, b, Hc)
        log(f"  平移候选 [{src}]: t=({tx_:.2f},{ty_:.2f}) resp={r_:.3f} → ZNCC={sc:.4f}")
        if sc > best_score:
            best, best_score = (Hc, th, s, (tx_, ty_), resp, sc), sc
    Hm, th, s, t, resp, zncc = best
    return Hm, th, s, t, resp, resp_t, zncc


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
    H_ncc, m0 = stv.fit_homography(pts[keep], disp[keep])
    h00, h01, h10, h11 = H_ncc[0, 0], H_ncc[0, 1], H_ncc[1, 0], H_ncc[1, 1]
    ang_true = float(np.degrees(np.arctan2(h10 - h01, h00 + h11)))
    log(f"NCC 真值角度 {ang_true:+.3f}°, 控制点 {int(keep.sum())}")

    results = {}

    # ---------- 归因实验：通用纹理合成图（验证实现本身）vs PCB 模板（验证内容）----------
    log("== 自标定：通用纹理合成图 ==")
    rngm = np.random.default_rng(7)
    gen = cv2.GaussianBlur(rngm.random((Hf, Wf)).astype(np.float32), (0, 0), 3.0)
    gen = ((gen - gen.min()) / (gen.max() - gen.min()) * 255).astype(np.uint8)
    t0 = time.time()
    ok_gen, resid_gen, *_ = calibrate(gen, Wf, Hf, "通用纹理")
    t_gen = time.time() - t0
    results["cal_generic"] = dict(ok=bool(ok_gen), resid=resid_gen, t=round(t_gen, 1))

    log("== 自标定：PCB 模板 ==")
    t0 = time.time()
    ok_pcb, resid_pcb, k_th, k_sl, k_tx, k_ty = calibrate(tpl, Wf, Hf, "PCB模板")
    t_cal = time.time() - t0
    cal = (ok_pcb, resid_pcb, k_th, k_sl, k_tx, k_ty)
    results["cal_pcb"] = dict(ok=bool(ok_pcb), resid=resid_pcb, t=round(t_cal, 1))

    if not ok_pcb:
        log("PCB 模板自标定未通过 → FM 结论仅记为'当前实现不可用'，跳过真实对")
        with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
            json.dump(dict(ang_true=ang_true, results=results), f, indent=2, ensure_ascii=False)
        lines = [f"真值角度 {ang_true:+.3f}°",
                 f"自标定 通用纹理: {'通过' if ok_gen else '未通过'} (resid={resid_gen and round(resid_gen, 2)}px)",
                 f"自标定 PCB模板: {'通过' if ok_pcb else '未通过'} (resid={resid_pcb and round(resid_pcb, 2)}px)",
                 "FM 真实对: 未运行（自标定未通过）"]
        with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines))
        return

    # ---------- 真实对 FM ----------
    log("== Fourier-Mellin 真实对 ==")
    t0 = time.time()
    H_fm, th, s, t, resp_p, resp_t, zncc = fm_full(tpl, smp, cal, Wf, Hf)
    t_fm = time.time() - t0
    dev_fm = dense_dev(H_fm, H_ncc, Wf, Hf)
    log(f"FM: dθ={th:+.3f}°(真{ang_true:+.3f}) s={s:.5f} t=({t[0]:.2f},{t[1]:.2f}), "
        f"ZNCC={zncc:.4f}, dev med={dev_fm['median']:.2f}px, {t_fm:.2f}s")
    results["fm"] = dict(theta=th, scale=s, t_vec=[float(t[0]), float(t[1])],
                         resp_polar=float(resp_p), resp_trans=float(resp_t), zncc=zncc,
                         dev=dev_fm, time_s=round(t_fm, 2))

    # ---------- FM → ECC 金字塔 ----------
    log("== FM 初值 → ECC ==")
    t0 = time.time()
    Hm = H_fm.copy()
    cc = -1.0
    for f in (4, 2, 1):
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
            break
        Hm = (np.diag([f, f, 1.0]) @ w.astype(np.float64) @ np.diag([1.0 / f, 1.0 / f, 1.0]))
        log(f"  ECC 1/{f}: cc={cc:.4f}")
    t_ecc = time.time() - t0
    if cc > 0:
        dev_ecc = dense_dev(Hm, H_ncc, Wf, Hf)
        log(f"FM+ECC: dev med={dev_ecc['median']:.2f} RMS={dev_ecc['rms']:.2f}px, {t_ecc:.1f}s")
        results["fm_ecc"] = dict(dev=dev_ecc, t=round(t_ecc, 1), cc=float(cc))

    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(dict(ang_true=ang_true, results=results), f, indent=2, ensure_ascii=False)

    # 汇总图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [("Fourier-Mellin(定标)", dev_fm)]
    if "fm_ecc" in results:
        rows.append(("FM→ECC", results["fm_ecc"]["dev"]))
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.18, [r[1]["median"] for r in rows], 0.36, label="median dev (px)")
    ax.bar(x + 0.18, [r[1]["rms"] for r in rows], 0.36, label="RMS dev (px)")
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=9)
    ax.set_ylabel("dense dev vs NCC-H (px @0.5x)")
    ax.grid(alpha=0.3, axis="y"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fm_dev.png"), dpi=130)
    plt.close(fig)

    lines = [f"真值角度 {ang_true:+.3f}°",
             f"自标定 通用纹理: {'通过' if ok_gen else '未通过'} (resid={resid_gen and round(resid_gen, 2)}px)",
             f"自标定 PCB模板: 通过 (resid={resid_pcb:.2f}px, k_th={k_th:+.0f} k_sl={k_sl:+.0f} k_t=({k_tx:+.0f},{k_ty:+.0f}))",
             f"FM: dθ={th:+.3f}° s={s:.5f} t=({t[0]:.2f},{t[1]:.2f}) ZNCC={zncc:.4f} "
             f"dev med={dev_fm['median']:.2f} RMS={dev_fm['rms']:.2f} P95={dev_fm['p95']:.2f}px, {t_fm:.2f}s",
             (f"FM→ECC: dev med={results['fm_ecc']['dev']['median']:.2f} RMS={results['fm_ecc']['dev']['rms']:.2f}px, "
              f"{results['fm_ecc']['t']}s, cc={results['fm_ecc']['cc']:.3f}") if "fm_ecc" in results else "FM→ECC: 失败"]
    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

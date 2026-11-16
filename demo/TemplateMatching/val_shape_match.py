# -*- coding: utf-8 -*-
"""
形状匹配在 2652s2g / 506s4e 上的补测（自研 Halcon 级实现，复用 halcon_shape_match.py）。

要点：
  - 倍角方向场（cos2θ/sin2θ）对灰度极性天然免疫 → 506s4e 跨域/极性问题不必取反
  - 自研实现只搜旋转+平移、无缩放 → 用 FM 先验的 scale 预缩放模板（绕自身中心）
  - 两图尺寸不同 → 各自居中 pad 到公共画布，形状匹配在画布系完成
  - 结果合成回"工作分辨率模板 → 采样"的相似变换 H_shape，用统一验收门评估
    （506s4e 跨域：亮度 NCC 天花板低，另给出边缘域验收参考）

用法：python val_shape_match.py <2652|506> <pair_no>
"""
import cv2
import numpy as np
import os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from halcon_shape_match import (hp, edge_points, fft_search, parabolic,
                                refine_progressive, EDGE_PCT, REFINE_PCT,
                                N_POINTS, HP_SIGMA, log)
from val_2652s2g import load_down, coarse_prior, accept, T
from make_overlay_heatmap import local_match

DATASETS = {"2652": r"D:\AOI\aoisys\2652s2g-gtl",
            "506": r"D:\AOI\aoisys\506s4e"}


def build_H(a_deg, tx, ty, c, ot, os_, s, c_t, w_t, h_t):
    """模板工作系 → 采样工作系 的相似变换：
    p --S(s;c_t)--> 缩放帧 --T(ot)--> pad --M(a,tx,ty;c)--> q_pad --T(-os)--> q"""
    S = np.eye(3)
    S[0, 0] = S[1, 1] = s
    S[0, 2] = c_t[0] * (1 - s)
    S[1, 2] = c_t[1] * (1 - s)
    a_ = np.radians(a_deg)
    R = np.array([[np.cos(a_), -np.sin(a_)], [np.sin(a_), np.cos(a_)]])
    M = np.eye(3)
    M[:2, :2] = R
    M[:2, 2] = np.array([tx, ty]) + c - R @ c
    H = T(-os_[0], -os_[1]) @ M @ T(ot[0], ot[1]) @ S
    return H


def shape_start_from_H0(H0, s, c_t, otp, os_, c):
    """从 FM 先验 H0（工作系模板→采样）解画布系形状匹配起步参数 (a°, tx, ty)。
    恒等式：build 链(H0) == 画布链(形状匹配)，解出 M 的平移：
      t_m = H0[:2,2] + os − c − R0·(ot' + (1−s)·c_t − c)，R0 = H0[:2,:2]/s
    （不能用 t_lowres×2 拼凑——预缩放后画布尺寸/中心全变，实测差 242px）"""
    R0 = H0[:2, :2] / s
    t_m = H0[:2, 2] + np.array(os_, float) - c - R0 @ (np.array(otp, float) + (1 - s) * c_t - c)
    a = np.degrees(np.arctan2(R0[1, 0] - R0[0, 1], R0[0, 0] + R0[1, 1]))
    return a, float(t_m[0]), float(t_m[1])


def run(base, ti, out, edge_accept=False):
    tpl = load_down(os.path.join(base, "template", f"template_{ti}.tif"), 8)
    smp = load_down(os.path.join(base, "sample", f"test_imgs{ti}.tif"), 8)
    h_t, w_t = tpl.shape

    # ---- FM 缩放先验（θ,s 稳健；t 不用）----
    H0, diag = coarse_prior(tpl, smp)
    s_est = float(diag.get("scale", 1.0))
    log(f"FM 先验: scale={s_est:.4f} dθ={diag.get('dtheta')}°")

    # ---- 模板按 s 预缩放（绕自身中心）----
    if abs(s_est - 1.0) > 0.005:
        tpl_s = cv2.resize(tpl, (int(round(w_t * s_est)), int(round(h_t * s_est))),
                           interpolation=cv2.INTER_AREA)
    else:
        tpl_s = tpl
    c_t = np.array([w_t / 2.0, h_t / 2.0])
    h_ts, w_ts = tpl_s.shape

    # ---- pad 居中公共画布 ----
    Hc, Wc = max(h_ts, smp.shape[0]), max(w_ts, smp.shape[1])
    ot = ((Wc - w_ts) // 2, (Hc - h_ts) // 2)
    os_ = ((Wc - smp.shape[1]) // 2, (Hc - smp.shape[0]) // 2)

    def pad(im, ox, oy):
        p = np.zeros((Hc, Wc), np.uint8)
        p[oy:oy + im.shape[0], ox:ox + im.shape[1]] = im
        return p

    tp, sp = pad(tpl_s, *ot), pad(smp, *os_)
    c = np.array([Wc / 2.0, Hc / 2.0])

    # ---- 形状匹配：FM 先验起步（重复板方向场全局搜索多峰，必须配粗定位）----
    tg = tp.astype(np.float32)
    sg = sp.astype(np.float32)
    tg_hp, sg_hp = hp(tg, HP_SIGMA), hp(sg, HP_SIGMA)
    # FM 先验 H0 → 画布系起步参数（s 已被模板预缩放吸收）
    a_sub, dx_sub, dy_sub = shape_start_from_H0(H0, s_est, c_t, ot, os_, c)
    log(f"FM 先验起步: a={a_sub:+.3f}° d=({dx_sub:.1f},{dy_sub:.1f})")

    # LSQ 精化（画布=工作分辨率，方向场多尺度；起点=FM 先验 ±12px 网格）
    log("LSQ 精化 ...")
    ex, ey, eth, ew = edge_points(tg, REFINE_PCT, N_POINTS)
    P2 = np.column_stack([ex, ey])
    gx = cv2.Sobel(sg, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(sg, cv2.CV_32F, 0, 1, 3)
    ang_map = cv2.phase(gx, gy).astype(np.float32)
    mag_map = cv2.magnitude(gx, gy).astype(np.float32)
    starts = [(a_sub, dx_sub + ox, dy_sub + oy) for ox in (-12, 0, 12) for oy in (-12, 0, 12)]
    t0 = time.time()
    a_fin, tx_fin, ty_fin, s_fin = refine_progressive(P2, eth, ew, sg, c, starts,
                                                      sigmas=(16.0, 8.0, 3.0, 0.0))
    log(f"  精化: a={a_fin:+.3f}° T=({tx_fin:.2f},{ty_fin:.2f}) score={s_fin:.1f} ({time.time() - t0:.1f}s)")

    # ---- 合成 H_shape 并评估 ----
    H_shape = build_H(a_fin, tx_fin, ty_fin, c, ot, os_, s_est, c_t, w_t, h_t)

    if not edge_accept:
        ok, m, pts1, err1 = accept(tpl, smp, H_shape)
        log(f"验收(亮度域): {'通过' if ok else '未通过'} {m}")
        summary = {"ok": bool(ok), **m}
        if len(err1):
            overlay_hm(tpl, smp, H_shape, out, ti, ok, m)
    else:
        # 跨域：边缘图局部匹配验收（结构域）
        t_e = edge_img(tpl)
        s_e = edge_img(smp)
        gx2 = np.arange(64, w_t - 64, 128, dtype=np.float64)
        gy2 = np.arange(64, h_t - 64, 128, dtype=np.float64)
        Xg, Yg = np.meshgrid(gx2, gy2)
        grid = np.stack([Xg.ravel(), Yg.ravel()], -1)
        wr = np.c_[grid, np.ones(len(grid))] @ H_shape.T
        mapped = wr[:, :2] / wr[:, 2:3]
        found, sc = local_match(t_e, s_e, grid, mapped, patch=64, search=20)
        okv = np.isfinite(found[:, 0]) & (sc >= 0.4)
        err = (np.linalg.norm(mapped[okv] - found[okv], axis=1) if okv.sum() else np.zeros(0))
        med = float(np.median(err)) if len(err) else -1
        ok = okv.mean() >= 0.3 and 0 < med <= 3.0
        log(f"验收(边缘域): 内点率 {okv.mean():.1%}, 残差 med {med:.2f}px, ZNCC med "
            f"{float(np.median(sc[okv])) if okv.sum() else -1:.3f} → {'通过' if ok else '未通过'}")
        summary = {"edge_inlier": float(okv.mean()), "edge_resid_med": med, "ok": bool(ok)}
        pts1, err1 = grid[okv], err
        if len(err1):
            overlay_hm(tpl, smp, H_shape, out, ti, ok, summary, edge=True)
    summary.update({"a_deg": a_fin, "t": [tx_fin, ty_fin], "score": s_fin,
                    "scale_est": s_est})
    return summary


def edge_img(img):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(32, 32))
    x = clahe.apply(img)
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, 3)
    e = cv2.magnitude(gx, gy)
    return np.clip(e, 0, np.percentile(e, 99)).astype(np.uint8)


def overlay_hm(tpl, smp, H, out, ti, ok, m, edge=False):
    Xf2, Yf2 = np.meshgrid(np.arange(tpl.shape[1], dtype=np.float32),
                           np.arange(tpl.shape[0], dtype=np.float32))
    wpts = np.c_[Xf2.ravel(), Yf2.ravel(), np.ones(Xf2.size, np.float32)] @ H.T.astype(np.float32)
    mxy = (wpts[:, :2] / wpts[:, 2:3]).astype(np.float32)
    warp = cv2.remap(smp, mxy[:, 0].reshape(tpl.shape), mxy[:, 1].reshape(tpl.shape),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    ov = np.zeros((tpl.shape[0], tpl.shape[1], 3), np.uint8)
    ov[..., 2] = tpl
    ov[..., 1] = warp
    s = 1400.0 / tpl.shape[1]
    cv2.imwrite(os.path.join(out, f"p{ti}_ov_shape.png"),
                cv2.resize(ov, (1400, int(tpl.shape[0] * s))))
    print(f"-> {os.path.join(out, f'p{ti}_ov_shape.png')} ok={ok} {m}")


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "2652"
    pair = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    out = os.path.join(HERE, f"val_{ {'2652': '2652s2g-gtl', '506': '506s4e'}[ds] }_output")
    os.makedirs(out, exist_ok=True)
    edge_accept = (ds == "506")
    res = run(DATASETS[ds], pair, out, edge_accept=edge_accept)
    import json
    with open(os.path.join(out, f"p{pair}_shape_match.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

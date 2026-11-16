# -*- coding: utf-8 -*-
"""
补测重跑 3：Farnebäck 加置信度掩码，并与 NCC+TPS 正面对比（形变估计器选型）。
置信度掩码 = 纹理（模板局部方差）× 前后向流一致性。
对比口径（留出控制点，与 TPS 训练互斥）：
  纯H / H+TPS / 稠密光流 在留出点上的位移误差（掩码内 vs 全部）；
  再加"翘曲后重跑 NCC 网格匹配"的残余位移与分数——对齐质量的独立判据。
输出：retest_farneback_tps_output/
"""
import cv2
import numpy as np
import os, time, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import synthetic_tps_validation as stv

OUT_DIR = os.path.join(HERE, "retest_farneback_tps_output")
os.makedirs(OUT_DIR, exist_ok=True)
DOWNSCALE = 0.5
BLOCK = 32          # 纹理掩码块大小
TEX_STD_MIN = 4.0   # 块内灰度 std 下限
CONSIST_THRES = 1.5  # 前后向流一致性阈值 (px)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def apply_h(Hm, pts):
    w = np.c_[pts, np.ones(len(pts))] @ Hm.T
    return w[:, :2] / w[:, 2:3]


def bilinear_sample(img, pts):
    """双线性采样 img 在 (x,y) 点的值（流场按通道）。"""
    h, w = img.shape[:2]
    x = np.clip(pts[:, 0], 0, w - 1.001)
    y = np.clip(pts[:, 1], 0, h - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = x - x0, y - y0
    if img.ndim == 2:
        return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy) +
                img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)
    out = np.zeros((len(pts), img.shape[2]))
    for c in range(img.shape[2]):
        ch = img[..., c]
        out[:, c] = (ch[y0, x0] * (1 - fx) * (1 - fy) + ch[y0, x0 + 1] * fx * (1 - fy) +
                     ch[y0 + 1, x0] * (1 - fx) * fy + ch[y0 + 1, x0 + 1] * fx * fy)
    return out


def postwarp_check(tpl, smp, map_x, map_y, tag):
    """按位移场 remap 后重跑 NCC 网格匹配。"""
    warped = cv2.remap(smp, map_x.astype(np.float32), map_y.astype(np.float32),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    pts, disp, sc = stv.match_grid(tpl, warped)
    d = np.linalg.norm(disp, axis=1)
    res = dict(med_resid=float(np.median(d)), p95_resid=float(np.percentile(d, 95)),
               med_ncc=float(np.median(sc)), n=int(len(d)))
    log(f"  {tag}: 重匹配 {res['n']} 点, 残余位移 med={res['med_resid']:.3f} P95={res['p95_resid']:.3f}px, "
        f"NCC med={res['med_ncc']:.4f}")
    return res, warped


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
    H_ncc, _ = stv.fit_homography(pts[keep], disp[keep])
    kp = pts[keep]; dp = disp[keep]
    log(f"NCC 控制点 {len(kp)}")

    # 留出划分（80/20，rng0，与 stv 惯例一致）
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(kp))
    n_tr = int(0.8 * len(kp))
    tr, te = idx[:n_tr], idx[n_tr:]
    log(f"训练 {len(tr)} / 留出 {len(te)}")

    # ---------- TPS（只用训练点）----------
    t0 = time.time()
    rbf, lam, rmse_te, tscale, rmse_h = stv.fit_tps(kp[tr], dp[tr], H_ncc)
    t_tps = time.time() - t0
    gated = rbf is None
    log(f"TPS: gated={gated} λ={lam}, {t_tps:.1f}s")

    # ---------- Farnebäck 前后向光流 ----------
    log("计算前向光流…")
    t0 = time.time()
    flow_f = cv2.calcOpticalFlowFarneback(tpl, smp, None, 0.5, 6, 41, 10, 7, 1.5,
                                          cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    t_f = time.time() - t0
    log(f"前向 {t_f:.1f}s; 计算后向…")
    t0 = time.time()
    flow_b = cv2.calcOpticalFlowFarneback(smp, tpl, None, 0.5, 6, 41, 10, 7, 1.5,
                                          cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    t_b = time.time() - t0
    log(f"后向 {t_b:.1f}s")

    # 前后向一致性: |flow_f(p) + flow_b(p+flow_f(p))|
    gy, gx = np.mgrid[0:Hf, 0:Wf].astype(np.float64)
    tgt_x = gx + flow_f[..., 0]
    tgt_y = gy + flow_f[..., 1]
    bx = bilinear_sample(flow_b[..., 0], np.c_[tgt_x.ravel(), tgt_y.ravel()]).reshape(Hf, Wf)
    by = bilinear_sample(flow_b[..., 1], np.c_[tgt_x.ravel(), tgt_y.ravel()]).reshape(Hf, Wf)
    consist = np.sqrt((flow_f[..., 0] + bx) ** 2 + (flow_f[..., 1] + by) ** 2)

    # 纹理掩码（模板块 std）
    th, tw = Hf // BLOCK, Wf // BLOCK
    tex = np.zeros((th, tw))
    for i in range(th):
        for j in range(tw):
            tex[i, j] = tpl[i * BLOCK:(i + 1) * BLOCK, j * BLOCK:(j + 1) * BLOCK].std()
    tex_ok = tex >= TEX_STD_MIN
    mask = np.kron(tex_ok, np.ones((BLOCK, BLOCK), bool))
    mask = np.pad(mask, ((0, Hf - mask.shape[0]), (0, Wf - mask.shape[1])), constant_values=False)[:Hf, :Wf]
    mask &= consist < CONSIST_THRES
    cov = float(mask.mean())
    log(f"掩码覆盖率 {cov:.1%}（纹理 {tex_ok.mean():.1%} × 一致性 {(consist < CONSIST_THRES).mean():.1%}）")

    # ---------- 留出点误差对比 ----------
    def err_flow(sel):
        fl = bilinear_sample(flow_f, kp[sel])
        e = np.linalg.norm(fl - dp[sel], axis=1)
        m = mask[np.clip(kp[sel][:, 1].astype(int), 0, Hf - 1), np.clip(kp[sel][:, 0].astype(int), 0, Wf - 1)]
        return e, m

    eH = np.linalg.norm(apply_h(H_ncc, kp[te]) - (kp[te] + dp[te]), axis=1)
    st_H = dict(median=float(np.median(eH)), rms=float(np.sqrt((eH ** 2).mean())), p95=float(np.percentile(eH, 95)))
    if not gated:
        resid_te = rbf(kp[te] / tscale)
        eT = np.linalg.norm(apply_h(H_ncc, kp[te]) + resid_te - (kp[te] + dp[te]), axis=1)
        st_T = dict(median=float(np.median(eT)), rms=float(np.sqrt((eT ** 2).mean())), p95=float(np.percentile(eT, 95)))
    else:
        st_T = None
    eF, mF = err_flow(te)
    st_F_all = dict(median=float(np.median(eF)), rms=float(np.sqrt((eF ** 2).mean())), p95=float(np.percentile(eF, 95)))
    st_F_mask = (dict(median=float(np.median(eF[mF])), rms=float(np.sqrt((eF[mF] ** 2).mean())),
                      p95=float(np.percentile(eF[mF], 95))),) if mF.sum() >= 5 else (None,)
    st_F_mask = st_F_mask[0]
    tps_txt = "(gated)" if st_T is None else f"med={st_T['median']:.3f}"
    fm_txt = "NA" if st_F_mask is None else f"med={st_F_mask['median']:.3f}"
    log(f"留出点误差  纯H: med={st_H['median']:.3f}  TPS: {tps_txt}  "
        f"flow(全部): med={st_F_all['median']:.3f}  flow(掩码内): {fm_txt}")

    # ---------- 翘曲后重匹配（对齐质量独立判据）----------
    results = dict(t_flow=[round(t_f, 1), round(t_b, 1)], t_tps=round(t_tps, 1), tps_gated=gated,
                   lambda_=lam, mask_cov=round(cov, 4), n_te=int(len(te)),
                   H_only=st_H, TPS=st_T, flow_all=st_F_all, flow_masked=st_F_mask)

    log("== 翘曲后重匹配对比 ==")
    # H 与 flow 场用全分辨率精确构造（注意：stride-8+resize 会因 OpenCV 半像素对齐错位数 px）
    Xf, Yf = np.meshgrid(np.arange(Wf, dtype=np.float32), np.arange(Hf, dtype=np.float32))
    wpts = np.c_[Xf.ravel(), Yf.ravel(), np.ones(Xf.size, np.float32)] @ H_ncc.T.astype(np.float32)
    hfull = (wpts[:, :2] / wpts[:, 2:3]).astype(np.float32)
    Hmx = hfull[:, 0].reshape(Hf, Wf)
    Hmy = hfull[:, 1].reshape(Hf, Wf)
    del wpts, hfull
    res_H, _ = postwarp_check(tpl, smp, Hmx, Hmy, "纯H")
    results["postwarp_H"] = res_H
    # H+TPS：TPS 残差是低频场，stride-8 采样 + resize 足够（RBF 全像素求值内存爆炸）
    if not gated:
        xs8 = np.arange(0, Wf, 8, dtype=np.float64)
        ys8 = np.arange(0, Hf, 8, dtype=np.float64)
        X8, Y8 = np.meshgrid(xs8, ys8)
        grid8 = np.stack([X8.ravel(), Y8.ravel()], axis=-1)
        rr8 = rbf(grid8 / tscale)
        ny8, nx8 = len(ys8), len(xs8)
        res_T, _ = postwarp_check(tpl, smp,
                                  Hmx + cv2.resize(rr8[:, 0].reshape(ny8, nx8).astype(np.float32), (Wf, Hf),
                                                   interpolation=cv2.INTER_LINEAR),
                                  Hmy + cv2.resize(rr8[:, 1].reshape(ny8, nx8).astype(np.float32), (Wf, Hf),
                                                   interpolation=cv2.INTER_LINEAR), "H+TPS")
        results["postwarp_TPS"] = res_T
    # 光流：全位移场 map = t + flow(t)（flow 本身就是全分辨率数组）
    res_F, _ = postwarp_check(tpl, smp, Xf + flow_f[..., 0].astype(np.float32),
                              Yf + flow_f[..., 1].astype(np.float32), "H+flow")
    results["postwarp_flow"] = res_F

    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 可视化：掩码 + 留出误差柱状图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    ov = cv2.cvtColor(cv2.resize(tpl, None, fx=0.4, fy=0.4), cv2.COLOR_GRAY2RGB)
    msmall = cv2.resize(mask.astype(np.uint8) * 255, (ov.shape[1], ov.shape[0]), interpolation=cv2.INTER_NEAREST)
    ov[msmall > 0] = ov[msmall > 0] * 0.4 + np.array([255, 80, 80]) * 0.6
    axes[0].imshow(ov); axes[0].set_title(f"confidence mask (cov {cov:.0%})"); axes[0].axis("off")
    names, meds = ["H"], [st_H["median"]]
    if st_T: names.append("H+TPS"); meds.append(st_T["median"])
    names.append("flow"); meds.append(st_F_all["median"])
    if st_F_mask: names.append("flow\nmasked"); meds.append(st_F_mask["median"])
    axes[1].bar(names, meds, color=["#888", "#2a7", "#e73", "#e73"][:len(names)])
    axes[1].set_ylabel("held-out ctrl err median (px)")
    axes[1].grid(alpha=0.3, axis="y")
    for i, v in enumerate(meds):
        axes[1].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "farneback_vs_tps.png"), dpi=130)
    plt.close(fig)

    def fmt(s):
        return "NA" if s is None else f"med={s['median']:.3f} RMS={s['rms']:.3f} P95={s['p95']:.3f}"
    res_T = results.get("postwarp_TPS")
    tps_pw_txt = "NA(gated)" if res_T is None else f"med={res_T['med_resid']:.3f}px NCC={res_T['med_ncc']:.4f}"
    lines = [f"控制点 {len(kp)}（训练 {len(tr)} / 留出 {len(te)}）, 掩码覆盖 {cov:.1%}, TPS gated={gated} λ={lam}",
             f"光流耗时 前向 {t_f:.1f}s + 后向 {t_b:.1f}s（TPS {t_tps:.1f}s）",
             f"留出点位移误差  纯H: {fmt(st_H)}",
             f"                H+TPS: {fmt(st_T)}",
             f"                flow全部: {fmt(st_F_all)}   flow掩码内: {fmt(st_F_mask)}",
             f"后配准残余  纯H: med={res_H['med_resid']:.3f}px NCC={res_H['med_ncc']:.4f}",
             f"            H+TPS: {tps_pw_txt}",
             f"            H+flow: med={res_F['med_resid']:.3f}px NCC={res_F['med_ncc']:.4f}"]
    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

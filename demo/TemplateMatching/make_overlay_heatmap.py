# -*- coding: utf-8 -*-
"""
各算法"对齐叠图 + 匹配误差热力图"图集（weights222 真实对 @0.5x）。

对每个算法输出两张图：
  1. 叠图 ov_*.png    —— 按该算法估计的变换把采样图翘曲回模板系，
                         红通道=模板、绿通道=翘曲采样，对齐处呈黄色，附中心放大窗。
  2. 热力图 hm_*.png  —— 匹配/对齐残差的空间分布（模板灰度底图 + turbo 色散点，
                         颜色=残差 px，统一截断 0~3px 便于跨算法对比；稠密场用面热力图）。

覆盖算法（9 类）：
  NCC-H、NCC+TPS(留出)、FM→ECC、SIFT、ORB、ALIKED+LG、SP+LG、LoFTR、Farnebäck 光流。
输出：overlay_heatmap_output/
"""
import cv2
import numpy as np
import os, time, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# lightglue 无 PyPI 包，使用 EfficientLoFTR 内的源码副本
sys.path.insert(0, r"D:\AOI\aoisys\aoisystem\TemplateMatching\EfficientLoFTR")
import synthetic_tps_validation as stv
from retest_phase_corr import calibrate, fm_full
from classical_feature_test import refine, mag_h
from make_match_gallery import overlay, local_verify

OUT = os.path.join(HERE, "overlay_heatmap_output")
os.makedirs(OUT, exist_ok=True)
DOWNSCALE = 0.5
VMAX = 3.0        # 热力图统一色标上限 px
TPL_PATH = r"D:\Users\Administrator\Desktop\weights222\object_region_full_res.jpg"
SMP_PATH = r"D:\Users\Administrator\Desktop\weights222\corresponding_sample_region_full_res.jpg"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def warp_by_H(smp, Hm, Wf, Hf):
    Xf, Yf = np.meshgrid(np.arange(Wf, dtype=np.float32), np.arange(Hf, dtype=np.float32))
    wpts = np.c_[Xf.ravel(), Yf.ravel(), np.ones(Xf.size, np.float32)] @ Hm.T.astype(np.float32)
    m = (wpts[:, :2] / wpts[:, 2:3]).astype(np.float32)
    return cv2.remap(smp, m[:, 0].reshape(Hf, Wf), m[:, 1].reshape(Hf, Wf),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def warp_by_field(smp, dx, dy):
    Xf, Yf = np.meshgrid(np.arange(smp.shape[1], dtype=np.float32),
                         np.arange(smp.shape[0], dtype=np.float32))
    return cv2.remap(smp, Xf + dx, Yf + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def heatmap_points(tpl, pts, err, path, title, vmax=VMAX):
    """散点热力图：模板灰度底图，散点颜色=残差。"""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    import matplotlib.pyplot as plt
    if len(pts) > 6000:
        idx = np.linspace(0, len(pts) - 1, 6000).astype(int)
        pts, err = pts[idx], err[idx]
    h, w = tpl.shape
    fig, ax = plt.subplots(figsize=(9, 9 * h / w))
    ax.imshow(tpl, cmap="gray", alpha=0.9)
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=np.minimum(err, vmax), cmap="turbo",
                    s=5, vmin=0, vmax=vmax, linewidths=0)
    med = np.median(err)
    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label(f"对齐残差 px（截断@{vmax:.0f}）")
    ax.set_title(f"{title}  med={med:.2f}px  p95={np.percentile(err, 95):.2f}px", fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log(f"  -> {os.path.basename(path)} (n={len(err)}, med={med:.2f}px)")


def heatmap_dense(err_grid, path, title, vmax):
    """稠密场热力图（光流 vs 全局 H 的局部残差）。"""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 9 * err_grid.shape[0] / err_grid.shape[1]))
    im = ax.imshow(np.minimum(err_grid, vmax), cmap="turbo", vmin=0, vmax=vmax,
                   extent=[0, err_grid.shape[1], err_grid.shape[0], 0])
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label(f"局部残差 px（截断@{vmax:.0f}）")
    med = np.median(err_grid)
    ax.set_title(f"{title}  med={med:.2f}px", fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log(f"  -> {os.path.basename(path)} (med={med:.2f}px)")


def local_match(tpl, smp, pts_t, pts_pred, patch=96, search=14, min_score=0.5):
    """模板 patch（pts_t 处）在采样图 pts_pred 附近局部搜索，返回 (命中位置, 分数)。"""
    h, w = tpl.shape
    pad = search + 2
    smp_pad = cv2.copyMakeBorder(smp, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    half = patch // 2
    found = np.full_like(pts_pred, np.nan)
    score = np.zeros(len(pts_t))
    for i, (pt_, pp) in enumerate(zip(pts_t, pts_pred)):
        tx, ty = int(round(pt_[0])), int(round(pt_[1]))
        px, py = int(round(pp[0])), int(round(pp[1]))
        x0t, y0t = tx - half, ty - half
        if x0t < 0 or y0t < 0 or x0t + patch > w or y0t + patch > h:
            continue
        if px - half < 0 or py - half < 0 or px + half > smp.shape[1] or py + half > smp.shape[0]:
            continue
        patch_t = tpl[y0t:y0t + patch, x0t:x0t + patch]
        if patch_t.std() < 4:
            continue
        win = smp_pad[py - half:py - half + patch + 2 * pad, px - half:px - half + patch + 2 * pad]
        r = cv2.matchTemplate(win, patch_t, cv2.TM_CCOEFF_NORMED)
        _, mv, _, mloc = cv2.minMaxLoc(r)
        score[i] = mv
        # win 在原图中的顶左 = (px-half-pad, py-half-pad)，patch 中心 = 顶左 + half
        found[i] = (px + mloc[0] - pad, py + mloc[1] - pad)
    return found, score


def feature_overlay_heatmap(name, tpl, smp, get_pairs, tag):
    """特征点类算法通用后处理：NCC 精化 -> MAGSAC -> 叠图 + 热力图。"""
    pt, ps = get_pairs()
    pr, nr = refine(tpl, smp, pt, ps)
    ok = nr >= 0.5
    H_ref, inl = mag_h(pt[ok], pr[ok])
    pi, qi = pt[ok][inl], pr[ok][inl]
    err = np.linalg.norm(qi - cv2.perspectiveTransform(
        pi.reshape(-1, 1, 2).astype(np.float32), H_ref).reshape(-1, 2), axis=1)
    warp = warp_by_H(smp, H_ref, tpl.shape[1], tpl.shape[0])
    overlay(tpl, warp, os.path.join(OUT, f"ov_{tag}.png"),
            f"{name}: warp by refined H (R=tpl G=sample)")
    heatmap_points(tpl, pi, err, os.path.join(OUT, f"hm_{tag}.png"),
                   f"{name} 对齐残差分布 ({int(inl.sum())} 内点)")
    return err


def main():
    log("读取图像…")
    tpl0 = cv2.imread(TPL_PATH, cv2.IMREAD_GRAYSCALE)
    smp0 = cv2.imread(SMP_PATH, cv2.IMREAD_GRAYSCALE)
    Wf, Hf = int(tpl0.shape[1] * DOWNSCALE), int(tpl0.shape[0] * DOWNSCALE)
    tpl = cv2.resize(tpl0, (Wf, Hf), interpolation=cv2.INTER_AREA)
    smp = cv2.resize(smp0, (Wf, Hf), interpolation=cv2.INTER_AREA)
    gallery = {}

    # ---------- 1. NCC pass1 (H) ----------
    log("== NCC pass1 ==")
    pts, disp, sc = stv.match_grid(tpl, smp)
    keep = stv.consistency_filter(pts, disp, sc)
    H_ncc, _ = stv.fit_homography(pts[keep], disp[keep])
    pred = cv2.perspectiveTransform(pts[keep].reshape(-1, 1, 2).astype(np.float32),
                                    H_ncc).reshape(-1, 2)
    err = np.linalg.norm(pts[keep] + disp[keep] - pred, axis=1)
    overlay(tpl, warp_by_H(smp, H_ncc, Wf, Hf), os.path.join(OUT, "ov_ncc_h.png"),
            "NCC pass1: warp by H (R=tpl G=sample)")
    heatmap_points(tpl, pts[keep], err, os.path.join(OUT, "hm_ncc_h.png"),
                   f"NCC-H 对齐残差分布 ({int(keep.sum())} 网格点)")
    gallery["NCC pass1 (H)"] = ("ov_ncc_h.png", "hm_ncc_h.png")

    # ---------- 2. NCC + TPS（留出 20%） ----------
    log("== NCC + TPS ==")
    from scipy.interpolate import RBFInterpolator
    kp, dp = pts[keep], disp[keep]
    rngs = np.random.default_rng(0)
    idxs = rngs.permutation(len(kp))
    ntr = int(0.8 * len(kp))
    rbf, lam, _, tscale, _ = stv.fit_tps(kp[idxs[:ntr]], dp[idxs[:ntr]], H_ncc)
    if rbf is not None:
        hold = idxs[ntr:]
        tps_d = rbf(kp[hold] / tscale)
        pred_t = cv2.perspectiveTransform(kp[hold].reshape(-1, 1, 2).astype(np.float32),
                                          H_ncc).reshape(-1, 2) + tps_d
        err_t = np.linalg.norm(kp[hold] + dp[hold] - pred_t, axis=1)
        # 全场翘曲（H + stride-8 TPS 残差）；必须用 meshgrid 行主序枚举，repeat/tile 列主序会搅乱映射
        Xf2, Yf2 = np.meshgrid(np.arange(Wf, dtype=np.float32), np.arange(Hf, dtype=np.float32))
        wpts = np.c_[Xf2.ravel(), Yf2.ravel(), np.ones(Xf2.size, np.float32)] @ H_ncc.T.astype(np.float32)
        hfull = (wpts[:, :2] / wpts[:, 2:3]).astype(np.float32)
        xs8 = np.arange(0, Wf, 8, dtype=np.float64)
        ys8 = np.arange(0, Hf, 8, dtype=np.float64)
        X8, Y8 = np.meshgrid(xs8, ys8)
        g8 = np.stack([X8.ravel(), Y8.ravel()], axis=-1)
        rr8 = rbf(g8 / tscale).astype(np.float32)
        ny8, nx8 = len(ys8), len(xs8)
        mx = hfull[:, 0].reshape(Hf, Wf) + cv2.resize(rr8[:, 0].reshape(ny8, nx8), (Wf, Hf),
                                                      interpolation=cv2.INTER_LINEAR)
        my = hfull[:, 1].reshape(Hf, Wf) + cv2.resize(rr8[:, 1].reshape(ny8, nx8), (Wf, Hf),
                                                      interpolation=cv2.INTER_LINEAR)
        warp_tps = cv2.remap(smp, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        overlay(tpl, warp_tps, os.path.join(OUT, "ov_ncc_tps.png"),
                "NCC+TPS: warp by H+TPS (R=tpl G=sample)")
        heatmap_points(tpl, kp[hold], err_t, os.path.join(OUT, "hm_ncc_tps.png"),
                       f"NCC+TPS 留出残差分布 ({len(hold)} 留出点)")
        gallery["NCC + TPS (留出)"] = ("ov_ncc_tps.png", "hm_ncc_tps.png")
    else:
        log("  TPS 被门控禁用，跳过")

    # ---------- 3. FM -> ECC ----------
    log("== FM -> ECC ==")
    cal = calibrate(tpl, Wf, Hf, "PCB")
    H_ecc = None
    if cal[0]:
        H_fm, *_ = fm_full(tpl, smp, cal, Wf, Hf)
        Hm = H_fm.copy()
        for f in (4, 2, 1):
            tw = cv2.resize(tpl, (Wf // f, Hf // f), interpolation=cv2.INTER_AREA)
            sw = cv2.resize(smp, (Wf // f, Hf // f), interpolation=cv2.INTER_AREA)
            S = np.diag([1.0 / f, 1.0 / f, 1.0])
            wmat = S @ Hm @ np.diag([f, f, 1.0])
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-7)
            try:
                _, wmat = cv2.findTransformECC(tw.astype(np.float32), sw.astype(np.float32),
                                               wmat.astype(np.float32).copy(),
                                               cv2.MOTION_HOMOGRAPHY, crit, None, 5)
            except cv2.error:
                break
            Hm = np.diag([f, f, 1.0]) @ wmat.astype(np.float64) @ np.diag([1.0 / f, 1.0 / f, 1.0])
        H_ecc = Hm
    if H_ecc is not None:
        step = 192
        gx = np.arange(step // 2, Wf - step // 2, step, dtype=np.float64)
        gy = np.arange(step // 2, Hf - step // 2, step, dtype=np.float64)
        X, Y = np.meshgrid(gx, gy)
        grid = np.stack([X.ravel(), Y.ravel()], axis=-1)
        wr = np.c_[grid, np.ones(len(grid))] @ H_ecc.T
        mapped = wr[:, :2] / wr[:, 2:3]
        found, sc2 = local_match(tpl, smp, grid, mapped)
        okv = np.isfinite(found[:, 0]) & (sc2 >= 0.5)
        err_f = np.linalg.norm(mapped[okv] - found[okv], axis=1)
        overlay(tpl, warp_by_H(smp, H_ecc, Wf, Hf), os.path.join(OUT, "ov_fm_ecc.png"),
                "FM->ECC: warp by global H (R=tpl G=sample)")
        heatmap_points(tpl, grid[okv], err_f, os.path.join(OUT, "hm_fm_ecc.png"),
                       f"FM->ECC 对齐残差分布 ({int(okv.sum())} 验证网格点)")
        gallery["FM->ECC 全局链路"] = ("ov_fm_ecc.png", "hm_fm_ecc.png")
    else:
        log("  FM/ECC 失败，跳过")

    # ---------- 4/5. SIFT / ORB ----------
    detectors = {"SIFT": cv2.SIFT_create(nfeatures=8000),
                 "ORB": cv2.ORB_create(nfeatures=8000, fastThreshold=10, patchSize=64)}
    for name, det in detectors.items():
        log(f"== {name} ==")
        kt, dt_ = det.detectAndCompute(tpl, None)
        ks, ds = det.detectAndCompute(smp, None)
        norm = cv2.NORM_HAMMING if name == "ORB" else cv2.NORM_L2
        pairs = cv2.BFMatcher(norm).knnMatch(dt_, ds, k=2)
        good = [m for m, n2 in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n2.distance]
        if len(good) < 10:
            log(f"  {name}: 匹配不足")
            continue
        pt = np.array([kt[m.queryIdx].pt for m in good], np.float64)
        ps = np.array([ks[m.trainIdx].pt for m in good], np.float64)
        err = feature_overlay_heatmap(name, tpl, smp, lambda: (pt, ps), name.lower())
        gallery[name] = (f"ov_{name.lower()}.png", f"hm_{name.lower()}.png")

    # ---------- 6. ALIKED + LightGlue ----------
    log("== ALIKED + LightGlue ==")
    try:
        import torch
        from lightglue import LightGlue
        from lightglue.aliked import ALIKED
        ext = ALIKED(max_num_keypoints=8000).eval()
        mat = LightGlue(features="aliked").eval()
        tt = lambda im: torch.from_numpy(im.astype(np.float32) / 255.0)[None, None]
        with torch.inference_mode():
            f_t, f_s = ext.extract(tt(tpl)), ext.extract(tt(smp))
            pred_ = mat({"image0": f_t, "image1": f_s})
            m = pred_["matches"][0].cpu().numpy()
            pt = f_t["keypoints"][0].cpu().numpy().astype(np.float64)[m[:, 0]]
            ps = f_s["keypoints"][0].cpu().numpy().astype(np.float64)[m[:, 1]]
        log(f"  LG 匹配 {len(m)} 对")
        err = feature_overlay_heatmap("ALIKED+LG", tpl, smp, lambda: (pt, ps), "aliked")
        gallery["ALIKED + LightGlue"] = ("ov_aliked.png", "hm_aliked.png")
    except Exception as e:
        log(f"  ALIKED 失败: {e}")

    # ---------- 7. SuperPoint + LightGlue ----------
    log("== SuperPoint + LightGlue ==")
    try:
        import torch
        from lightglue import LightGlue
        from lightglue.superpoint import SuperPoint
        ext = SuperPoint(max_num_keypoints=8000).eval()
        mat = LightGlue(features="superpoint").eval()
        tt = lambda im: torch.from_numpy(im.astype(np.float32) / 255.0)[None, None]
        with torch.inference_mode():
            f_t, f_s = ext.extract(tt(tpl)), ext.extract(tt(smp))
            pred_ = mat({"image0": f_t, "image1": f_s})
            m = pred_["matches"][0].cpu().numpy()
            pt = f_t["keypoints"][0].cpu().numpy().astype(np.float64)[m[:, 0]]
            ps = f_s["keypoints"][0].cpu().numpy().astype(np.float64)[m[:, 1]]
        log(f"  LG 匹配 {len(m)} 对")
        err = feature_overlay_heatmap("SP+LG", tpl, smp, lambda: (pt, ps), "superpoint")
        gallery["SuperPoint + LightGlue"] = ("ov_superpoint.png", "hm_superpoint.png")
    except Exception as e:
        log(f"  SuperPoint 失败: {e}")

    # ---------- 8. LoFTR ----------
    log("== LoFTR ==")
    try:
        import torch
        import kornia as K
        from kornia.feature import LoFTR
        LONG_SIDE = 1600
        sc_ = LONG_SIDE / max(Wf, Hf)
        lw, lh = int(Wf * sc_) // 8 * 8, int(Hf * sc_) // 8 * 8
        t_l = cv2.resize(tpl, (lw, lh), interpolation=cv2.INTER_AREA)
        s_l = cv2.resize(smp, (lw, lh), interpolation=cv2.INTER_AREA)
        matcher = LoFTR(pretrained="outdoor").eval()
        img0 = torch.from_numpy(t_l.astype(np.float32) / 255.0)[None, None]
        img1 = torch.from_numpy(s_l.astype(np.float32) / 255.0)[None, None]
        t0 = time.time()
        with torch.inference_mode():
            resp = matcher({"image0": img0, "image1": img1})
        log(f"  推理 {time.time() - t0:.1f}s")
        kp0, kp1 = resp["keypoints0"].cpu().numpy(), resp["keypoints1"].cpu().numpy()
        conf = resp.get("confidence")
        if conf is not None:
            cf = conf.cpu().numpy()
            mk = cf >= 0.5
            kp0, kp1 = kp0[mk], kp1[mk]
        sx, sy = Wf / lw, Hf / lh
        pt = np.column_stack([kp0[:, 0] * sx, kp0[:, 1] * sy])
        ps = np.column_stack([kp1[:, 0] * sx, kp1[:, 1] * sy])
        log(f"  匹配 {len(pt)} 对")
        err = feature_overlay_heatmap("LoFTR", tpl, smp, lambda: (pt, ps), "loftr")
        gallery["LoFTR"] = ("ov_loftr.png", "hm_loftr.png")
    except Exception as e:
        log(f"  LoFTR 失败: {e}")

    # ---------- 9. Farnebäck 光流（稠密残差热力图） ----------
    log("== Farnebäck 光流 ==")
    flow = cv2.calcOpticalFlowFarneback(tpl, smp, None, 0.5, 6, 41, 10, 7, 1.5,
                                        cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    warp_fl = warp_by_field(smp, flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32))
    overlay(tpl, warp_fl, os.path.join(OUT, "ov_flow.png"),
            "Farneback flow: warp by dense flow (R=tpl G=sample)")
    # 局部残差 = flow − 全局 H 预测位移（stride 8）
    st8 = 8
    fgx, fgy = (np.arange(0, Wf, st8), np.arange(0, Hf, st8))
    # 行主序枚举（与 flow[::8,::8].reshape 一致）；repeat/tile 列主序会使残差位置错乱
    Xg, Yg = np.meshgrid(fgx.astype(np.float64), fgy.astype(np.float64))
    P = np.stack([Xg.ravel(), Yg.ravel()], -1)
    pred_d = cv2.perspectiveTransform(P.reshape(-1, 1, 2).astype(np.float32),
                                      H_ncc).reshape(-1, 2) - P
    fv = flow[::st8, ::st8].reshape(-1, 2)
    err_d = np.linalg.norm(fv - pred_d, axis=1).reshape(len(fgy), len(fgx))
    # 稠密场残差远大于点残差（局部形变 med~9px），统一 3px 色标会全图饱和，改自适应
    vmax_d = float(max(5.0, np.percentile(err_d, 95)))
    heatmap_dense(err_d, os.path.join(OUT, "hm_flow.png"),
                  "Farneback 光流局部残差分布（扣除全局 H）", vmax_d)
    gallery["Farnebäck 光流"] = ("ov_flow.png", "hm_flow.png")

    # ---------- 总览拼图 ----------
    log("== 总览拼图 ==")
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    items = [(t, os.path.join(OUT, o), os.path.join(OUT, h)) for t, (o, h) in gallery.items()
             if os.path.exists(os.path.join(OUT, o)) and os.path.exists(os.path.join(OUT, h))]
    n = len(items)
    fig, axes = plt.subplots(n, 2, figsize=(16, 4.6 * n))
    axes = np.atleast_2d(axes)
    for k, (title, po, ph) in enumerate(items):
        ax_o, ax_h = axes[k, 0], axes[k, 1]
        ax_o.imshow(mpimg.imread(po))
        ax_o.set_title(f"{title} — 对齐叠图", fontsize=12)
        ax_o.axis("off")
        ax_h.imshow(mpimg.imread(ph))
        ax_h.set_title(f"{title} — 误差热力图", fontsize=12)
        ax_h.axis("off")
    fig.suptitle("各算法 对齐叠图 + 匹配误差热力图（weights222 @0.5x；点热力图色标 0~3px，光流稠密场自适应）", fontsize=15)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "overlay_heatmap_overview.png"), dpi=100)
    plt.close(fig)
    log(f"完成 -> overlay_heatmap_output/ ({n} 类方法)")


if __name__ == "__main__":
    main()

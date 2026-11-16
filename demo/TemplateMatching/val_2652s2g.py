# -*- coding: utf-8 -*-
"""
2652s2g-gtl 新样本全算法验证。

数据：D:\\AOI\\aoisys\\2652s2g-gtl
  template/template_i.tif  (13508~14616)x40985, 8bit 灰度
  sample/test_imgsi.tif    16384x47068 (第4对 7042x47068)
  配对假设：template_i ↔ test_imgsi（按序号）
工作分辨率：1/8（模板 ~1690x5123），先验级粗匹配再降 2 倍（原 1/16）。

流程（每对）：
  A 先验级   低分辨率 FM 自动粗匹配 → H0 → 统一验收门打分（验证"先验"本身）
  B 精配级   NCC 三遍粗到细(H0 锚定) → H_ncc → 验收 → 残差 TPS(80/20 留出门控)
  C 横向对比 FM→ECC / SIFT / ORB / ALIKED+LG / SP+LG / LoFTR（原始对）
             FM→ECC 与 Farnebäck 在 H_ncc 对齐对上跑（二者要求同尺寸输入）
  D 产物     ov_*/hm_* 叠图+热力图（val_2652s2g_output/，p{i}_ 前缀）+ summary JSON

用法：python val_2652s2g.py <pair_no 1..4 | all> [dataset_base] [invert_tpl]
  dataset_base 缺省为 2652s2g-gtl；传入其他同构数据集目录（含 template/ sample/ 子目录）
  时自动切换，输出目录为 val_{目录名}_output/（如 506s4e → val_506s4e_output/）。
  invert_tpl=1 时模板读取后立即取反（255-x）——用于"模板极性与采样相反"的料号
  （如 506s4e：模板白色=铜面、黑色=基板，与实拍采样极性相反）。
  注意：FM 幅度谱对极性不敏感，但平移相位相关与 NCC 在反相下会选错峰/给负分，
  因此必须在进入任何匹配前统一极性。
"""
import cv2
import numpy as np
import os, sys, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import synthetic_tps_validation as stv
from retest_phase_corr import calibrate, fm_full
from classical_feature_test import refine, mag_h
from make_overlay_heatmap import (overlay, heatmap_points, heatmap_dense,
                                  local_match, warp_by_H, warp_by_field)

BASE = r"D:\AOI\aoisys\2652s2g-gtl"
OUT = os.path.join(HERE, "val_2652s2g_output")
os.makedirs(OUT, exist_ok=True)
WORK = 8          # 工作分辨率 1/8
COARSE = 2        # 先验级在工作分辨率上再降 2 倍（即原图 1/16）
PAIRS = [(1, 1), (2, 2), (3, 3), (4, 4)]
LONG_SIDE = 1600  # LoFTR 输入

_DL_CACHE = {}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_down(path, scale):
    """读大 TIF 并降采样（raw 未压缩，整载后 INTER_AREA）。"""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(path)
    im.load()
    arr = np.ascontiguousarray(np.asarray(im))
    h, w = arr.shape
    out = cv2.resize(arr, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    del arr, im
    return out


def T(dx, dy):
    M = np.eye(3)
    M[0, 2], M[1, 2] = dx, dy
    return M


def coarse_prior(tpl_w, smp_w):
    """先验级：工作分辨率再降 COARSE 倍 → 居中 pad 到同尺寸 → FM 测 (θ,s,t)。
    返回 (H0 或 None, diag)：H0 把工作分辨率模板坐标映射到工作分辨率采样坐标。"""
    h_t, w_t = tpl_w.shape
    h_s, w_s = smp_w.shape
    t_l = cv2.resize(tpl_w, (w_t // COARSE, h_t // COARSE), interpolation=cv2.INTER_AREA)
    s_l = cv2.resize(smp_w, (w_s // COARSE, h_s // COARSE), interpolation=cv2.INTER_AREA)
    Wc, Hc = max(t_l.shape[1], s_l.shape[1]), max(t_l.shape[0], s_l.shape[0])

    def pad_center(img):
        oy, ox = (Hc - img.shape[0]) // 2, (Wc - img.shape[1]) // 2
        p = np.zeros((Hc, Wc), np.uint8)
        p[oy:oy + img.shape[0], ox:ox + img.shape[1]] = img
        return p, (ox, oy)

    tp, (otx, oty) = pad_center(t_l)
    sp, (osx, osy) = pad_center(s_l)
    cal = calibrate(tp, Wc, Hc, "coarse")
    if not cal[0]:
        return None, {"reason": "粗级自标定失败"}
    H_pad, th, s_, t_, resp, resp_t, zncc = fm_full(tp, sp, cal, Wc, Hc)
    D = np.diag([1.0 * COARSE, 1.0 * COARSE, 1.0])
    Di = np.diag([1.0 / COARSE, 1.0 / COARSE, 1.0])
    # p_w --Di(÷C)--> p_l --T(ot)--> pad --H_pad--> q_pad --T(-C·os)--> q_w
    # 注意输入侧必须先 ÷C（漏掉会使锚点整体偏离 C 倍）
    H0 = T(-COARSE * osx, -COARSE * osy) @ D @ H_pad @ T(otx, oty) @ Di
    return H0, {"dtheta": round(th, 4), "scale": round(s_, 5),
                "t_lowres": [round(v, 2) for v in t_], "zncc": round(zncc, 4),
                "resp": round(resp, 3)}


def accept(tpl, smp, Hm, step=128):
    """统一验收门（与 fallback_chain 相同口径）。返回 (ok, m, 验收点, 验收误差, ...)。"""
    gx = np.arange(step // 2, tpl.shape[1] - step // 2, step, dtype=np.float64)
    gy = np.arange(step // 2, tpl.shape[0] - step // 2, step, dtype=np.float64)
    Xg, Yg = np.meshgrid(gx, gy)                       # 行主序
    grid = np.stack([Xg.ravel(), Yg.ravel()], axis=-1)
    wr = np.c_[grid, np.ones(len(grid))] @ Hm.T
    mapped = wr[:, :2] / wr[:, 2:3]
    found, sc = local_match(tpl, smp, grid, mapped)
    ok = np.isfinite(found[:, 0]) & (sc >= 0.5)
    if ok.mean() < 0.3:
        return False, {"inlier_ratio": float(ok.mean())}, np.zeros((0, 2)), np.zeros(0)
    err = np.linalg.norm(mapped[ok] - found[ok], axis=1)
    m = {"inlier_ratio": float(ok.mean()), "resid_med": float(np.median(err)),
         "resid_p95": float(np.percentile(err, 95)), "zncc_med": float(np.median(sc[ok]))}
    return m["resid_med"] <= 3.0 and m["zncc_med"] >= 0.6, m, grid[ok], err


def ncc_three_pass(tpl, smp, H0):
    """三遍粗到细，返回 (H, pts, disp)（第三遍保留点，供 TPS 拟合）。"""
    Hm = H0
    pts_f = disp_f = None
    for srch, patch, ncc_min in [(stv.SEARCH_R, stv.PATCH, stv.NCC_MIN),
                                 (48, stv.PATCH, stv.NCC_MIN),
                                 (stv.SMALL_SEARCH, stv.SMALL_PATCH, stv.SMALL_NCC)]:
        pts, disp, sc = stv.match_grid(tpl, smp, H0=Hm, search_r=srch, patch=patch)
        keep = stv.consistency_filter(pts, disp, sc, ncc_min=ncc_min)
        if keep.sum() < 30:
            return None, None, None
        pts_f, disp_f = pts[keep], disp[keep]
        Hm, _ = stv.fit_homography(pts_f, disp_f)
    return Hm, pts_f, disp_f


def get_dl(name):
    """DL 模型懒加载单例（CPU）。"""
    if name in _DL_CACHE:
        return _DL_CACHE[name]
    import torch
    from lightglue import LightGlue
    if name == "aliked":
        from lightglue.aliked import ALIKED
        pair = (ALIKED(max_num_keypoints=8000).eval(), LightGlue(features="aliked").eval())
    else:
        from lightglue.superpoint import SuperPoint
        pair = (SuperPoint(max_num_keypoints=8000).eval(), LightGlue(features="superpoint").eval())
    _DL_CACHE[name] = pair
    return pair


def dl_match(name, tpl, smp):
    import torch
    ext, mat = get_dl(name)
    tt = lambda im: torch.from_numpy(im.astype(np.float32) / 255.0)[None, None]
    with torch.inference_mode():
        f_t, f_s = ext.extract(tt(tpl)), ext.extract(tt(smp))
        pred_ = mat({"image0": f_t, "image1": f_s})
        m = pred_["matches"][0].cpu().numpy()
        pt = f_t["keypoints"][0].cpu().numpy().astype(np.float64)[m[:, 0]]
        ps = f_s["keypoints"][0].cpu().numpy().astype(np.float64)[m[:, 1]]
    return pt, ps


def loftr_match(tpl, smp):
    import torch
    from kornia.feature import LoFTR
    if "loftr" not in _DL_CACHE:
        _DL_CACHE["loftr"] = LoFTR(pretrained="outdoor").eval()
    matcher = _DL_CACHE["loftr"]
    h, w = tpl.shape
    sc_ = LONG_SIDE / max(w, h)
    lw, lh = int(w * sc_) // 8 * 8, int(h * sc_) // 8 * 8
    t_l = cv2.resize(tpl, (lw, lh), interpolation=cv2.INTER_AREA)
    s_l = cv2.resize(smp, (lw, lh), interpolation=cv2.INTER_AREA)
    img0 = torch.from_numpy(t_l.astype(np.float32) / 255.0)[None, None]
    img1 = torch.from_numpy(s_l.astype(np.float32) / 255.0)[None, None]
    with torch.inference_mode():
        resp = matcher({"image0": img0, "image1": img1})
    kp0, kp1 = resp["keypoints0"].cpu().numpy(), resp["keypoints1"].cpu().numpy()
    conf = resp.get("confidence")
    if conf is not None:
        mk = conf.cpu().numpy() >= 0.5
        kp0, kp1 = kp0[mk], kp1[mk]
    sx, sy = w / lw, h / lh
    pt = np.column_stack([kp0[:, 0] * sx, kp0[:, 1] * sy])
    ps = np.column_stack([kp1[:, 0] * sx, kp1[:, 1] * sy])
    if len(pt) < 10:
        raise RuntimeError(f"LoFTR 有效匹配不足 ({len(pt)})")
    return pt, ps


def feature_block(tag, name, tpl, smp, pt, ps, summary):
    """特征类通用后处理：NCC 亚像素精化 → MAGSAC → 叠图 + 热力图。"""
    pr, nr = refine(tpl, smp, pt, ps)
    ok = nr >= 0.5
    H, inl = mag_h(pt[ok], pr[ok])
    pi, qi = pt[ok][inl], pr[ok][inl]
    err = np.linalg.norm(qi - cv2.perspectiveTransform(
        pi.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2), axis=1)
    overlay(tpl, warp_by_H(smp, H, tpl.shape[1], tpl.shape[0]),
            os.path.join(OUT, f"{tag}_ov_{name}.png"), f"{name}: warp by H (R=tpl G=sample)")
    heatmap_points(tpl, pi, err, os.path.join(OUT, f"{tag}_hm_{name}.png"),
                   f"{name} 对齐残差分布 ({int(inl.sum())} 内点)")
    summary[name] = {"n": int(inl.sum()), "resid_med": float(np.median(err)),
                     "resid_rms": float(np.sqrt((err ** 2).mean()))}
    log(f"  {name}: 内点 {int(inl.sum())}, med {np.median(err):.2f}px")


def validate_pair(ti, si, invert_tpl=False):
    tag = f"p{ti}"
    summary = {"pair": ti, "work_scale": WORK, "invert_tpl": invert_tpl}
    log(f"======== pair {ti}: template_{ti} ↔ test_imgs{si} ========")
    t0 = time.time()
    tpl = load_down(os.path.join(BASE, "template", f"template_{ti}.tif"), WORK)
    smp = load_down(os.path.join(BASE, "sample", f"test_imgs{si}.tif"), WORK)
    if invert_tpl:
        tpl = 255 - tpl          # 极性统一：必须在一切匹配之前（FM 幅度谱不受影响）
    log(f"工作分辨率 tpl {tpl.shape[1]}x{tpl.shape[0]}, smp {smp.shape[1]}x{smp.shape[0]} "
        f"(读取+降采样 {time.time() - t0:.1f}s)")

    # ---- A 先验级 ----
    t0 = time.time()
    H0, diag_c = coarse_prior(tpl, smp)
    log(f"先验级 FM: {diag_c} ({time.time() - t0:.1f}s)")
    summary["coarse_prior"] = diag_c
    if H0 is None:
        summary["error"] = "先验级失败"
        return summary
    ok0, m0, *_ = accept(tpl, smp, H0)
    log(f"先验级验收: {'通过' if ok0 else '未通过'} {m0}")
    summary["prior_accept"] = m0
    # 先验叠图
    overlay(tpl, warp_by_H(smp, H0, tpl.shape[1], tpl.shape[0]),
            os.path.join(OUT, f"{tag}_ov_prior.png"), "Prior (coarse FM): warp by H0")

    # ---- B1 精配级：FM 先验矫正 → ECC 金字塔精化（恒等初值）----
    # 该对存在 ~9% 真实缩放差：纯平移 patch NCC 在原始对上先天失效。
    # 两步法：先用 H0 矫正采样图（消除缩放/旋转，模板尺寸内），
    # 再在对齐对上用 ECC（8-dof，恒等初值）精化残余——
    # 直接在原始对上从 H0 起 ECC 会被采样图多余边幅带偏（实测失败）。
    Wt, Ht = tpl.shape[1], tpl.shape[0]
    t0 = time.time()
    smp_al0 = warp_by_H(smp, H0, Wt, Ht)
    Hm = np.eye(3)
    for f in (4, 2, 1):
        tw = cv2.resize(tpl, (Wt // f, Ht // f), interpolation=cv2.INTER_AREA)
        sw = cv2.resize(smp_al0, (Wt // f, Ht // f), interpolation=cv2.INTER_AREA)
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
    H_rel = Hm
    ok1, m1, pts1, err1 = accept(tpl, smp_al0, H_rel)
    log(f"FM先验→ECC 精化(矫正帧): {time.time() - t0:.1f}s, 验收{'通过' if ok1 else '未通过'} {m1}")
    summary["ecc_refine"] = m1
    if len(err1):
        overlay(tpl, warp_by_H(smp_al0, H_rel, Wt, Ht),
                os.path.join(OUT, f"{tag}_ov_fm_ecc.png"), "FM prior + ECC: warp by H0·H_rel")
        heatmap_points(tpl, pts1, err1, os.path.join(OUT, f"{tag}_hm_fm_ecc.png"),
                       f"FM->ECC 残差分布 ({len(pts1)} 验收点, 矫正帧)")
    if not ok1:
        # ---- B1b 先验失效兜底：ALIKED+LightGlue 无先验匹配 → 矫正 → ECC 精化 ----
        log("先验级+ECC 未过验收 → 切换 ALIKED+LG 无先验兜底")
        try:
            t0 = time.time()
            pt, ps = dl_match("aliked", tpl, smp)
            pr, nr = refine(tpl, smp, pt, ps)
            okr = nr >= 0.5
            H_al, inl_al = mag_h(pt[okr], pr[okr])
            smp_al0 = warp_by_H(smp, H_al, Wt, Ht)
            Hm = np.eye(3)
            for f in (4, 2, 1):
                tw = cv2.resize(tpl, (Wt // f, Ht // f), interpolation=cv2.INTER_AREA)
                sw = cv2.resize(smp_al0, (Wt // f, Ht // f), interpolation=cv2.INTER_AREA)
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
            H_rel = Hm
            ok1, m1, pts1, err1 = accept(tpl, smp_al0, H_rel)
            log(f"ALIKED→ECC 精化: {time.time() - t0:.1f}s, 验收{'通过' if ok1 else '未通过'} {m1}")
            summary["aliked_ecc"] = m1
            if len(err1):
                overlay(tpl, warp_by_H(smp_al0, H_rel, Wt, Ht),
                        os.path.join(OUT, f"{tag}_ov_aliked_ecc.png"),
                        "ALIKED fallback + ECC: warp by H_al·H_rel")
                heatmap_points(tpl, pts1, err1, os.path.join(OUT, f"{tag}_hm_aliked_ecc.png"),
                               f"ALIKED→ECC 残差分布 ({len(pts1)} 验收点)")
            if not ok1:
                summary["error"] = "ECC 精化未通过验收（ALIKED 兜底仍失败）"
                return summary
        except Exception as e:
            summary["error"] = f"ALIKED 兜底失败: {e}"
            return summary

    # ---- B2 对齐后 NCC 三遍 + 残差 TPS（对齐对 smp_al，恒等锚点）----
    smp_al = warp_by_H(smp_al0, H_rel, Wt, Ht)
    overlay(tpl, smp_al, os.path.join(OUT, f"{tag}_ov_aligned_ref.png"), "Reference: warp by FM->ECC H")
    t0 = time.time()
    H_rel, pts, disp = ncc_three_pass(tpl, smp_al, None)
    if H_rel is None:
        summary["error"] = "对齐后 NCC 三遍失败"
        return summary
    log(f"对齐后 NCC 三遍: {time.time() - t0:.1f}s, 最终保留 {len(pts)} 点")
    ok2, m2, *_ = accept(tpl, smp_al, H_rel)
    log(f"NCC-H(对齐后) 验收: {'通过' if ok2 else '未通过'} {m2}")
    summary["ncc_h"] = m2
    pred = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32),
                                    H_rel).reshape(-1, 2)
    err_h = np.linalg.norm(pts + disp - pred, axis=1)
    overlay(tpl, warp_by_H(smp_al, H_rel, Wt, Ht),
            os.path.join(OUT, f"{tag}_ov_ncc_h.png"), "NCC 3-pass (aligned pair): warp by H_rel")
    heatmap_points(tpl, pts, err_h, os.path.join(OUT, f"{tag}_hm_ncc_h.png"),
                   f"NCC-H 残差分布 ({len(pts)} 点, 对齐后)")

    # TPS（80/20 留出）
    from scipy.interpolate import RBFInterpolator
    rngs = np.random.default_rng(0)
    idxs = rngs.permutation(len(pts))
    ntr = int(0.8 * len(pts))
    rbf, lam, hold_rmse, tscale, hold_rmse_h = stv.fit_tps(pts[idxs[:ntr]], disp[idxs[:ntr]], H_rel)
    if rbf is not None:
        hold = idxs[ntr:]
        pred_t = cv2.perspectiveTransform(pts[hold].reshape(-1, 1, 2).astype(np.float32),
                                          H_rel).reshape(-1, 2) + rbf(pts[hold] / tscale)
        err_t = np.linalg.norm(pts[hold] + disp[hold] - pred_t, axis=1)
        heatmap_points(tpl, pts[hold], err_t, os.path.join(OUT, f"{tag}_hm_ncc_tps.png"),
                       f"NCC+TPS 留出残差 ({len(hold)} 点)")
        log(f"TPS 门控: {'启用' if rbf is not None else '禁用'} (留出 TPS {hold_rmse:.2f} vs H {hold_rmse_h:.2f}px)")
        summary["ncc_tps"] = {"holdout_med": float(np.median(err_t)), "lam": lam,
                              "hold_rmse_tps": float(hold_rmse), "hold_rmse_h": float(hold_rmse_h)}
        # TPS 叠图（H + stride-8 残差）
        Xf2, Yf2 = np.meshgrid(np.arange(tpl.shape[1], dtype=np.float32),
                               np.arange(tpl.shape[0], dtype=np.float32))
        wpts = np.c_[Xf2.ravel(), Yf2.ravel(), np.ones(Xf2.size, np.float32)] @ H_rel.T.astype(np.float32)
        hfull = (wpts[:, :2] / wpts[:, 2:3]).astype(np.float32)
        xs8 = np.arange(0, tpl.shape[1], 8, dtype=np.float64)
        ys8 = np.arange(0, tpl.shape[0], 8, dtype=np.float64)
        X8, Y8 = np.meshgrid(xs8, ys8)
        rr8 = rbf(np.stack([X8.ravel(), Y8.ravel()], -1) / tscale).astype(np.float32)
        ny8, nx8 = len(ys8), len(xs8)
        mx = hfull[:, 0].reshape(tpl.shape) + cv2.resize(rr8[:, 0].reshape(ny8, nx8),
                                                         (tpl.shape[1], tpl.shape[0]), interpolation=cv2.INTER_LINEAR)
        my = hfull[:, 1].reshape(tpl.shape) + cv2.resize(rr8[:, 1].reshape(ny8, nx8),
                                                         (tpl.shape[1], tpl.shape[0]), interpolation=cv2.INTER_LINEAR)
        warp_tps = cv2.remap(smp_al, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        overlay(tpl, warp_tps, os.path.join(OUT, f"{tag}_ov_ncc_tps.png"), "NCC+TPS (aligned pair)")
    else:
        log("TPS 被门控禁用")
        summary["ncc_tps"] = {"gated_off": True}

    # ---- C 横向对比（SIFT/ORB/DL 用原始对；FM→ECC 与 Farnebäck 已在对齐对上完成）----

    # SIFT / ORB（原始对）
    dets = {"SIFT": cv2.SIFT_create(nfeatures=8000),
            "ORB": cv2.ORB_create(nfeatures=8000, fastThreshold=10, patchSize=64)}
    for name, det in dets.items():
        try:
            kt, dt_ = det.detectAndCompute(tpl, None)
            ks, ds = det.detectAndCompute(smp, None)
            norm = cv2.NORM_HAMMING if name == "ORB" else cv2.NORM_L2
            pairs_ = cv2.BFMatcher(norm).knnMatch(dt_, ds, k=2)
            good = [m for m, n2 in (p for p in pairs_ if len(p) == 2)
                    if m.distance < 0.75 * n2.distance]
            if len(good) < 10:
                summary[name] = {"error": f"匹配不足 ({len(good)})"}
                log(f"  {name}: 匹配不足 ({len(good)})")
                continue
            pt = np.array([kt[m.queryIdx].pt for m in good], np.float64)
            ps = np.array([ks[m.trainIdx].pt for m in good], np.float64)
            feature_block(tag, name.lower(), tpl, smp, pt, ps, summary)
        except Exception as e:
            summary[name] = {"error": str(e)}
            log(f"  {name} 失败: {e}")

    # ALIKED / SP / LoFTR
    try:
        pt, ps = dl_match("aliked", tpl, smp)
        feature_block(tag, "aliked", tpl, smp, pt, ps, summary)
    except Exception as e:
        summary["ALIKED"] = {"error": str(e)}
        log(f"  ALIKED 失败: {e}")
    try:
        pt, ps = dl_match("superpoint", tpl, smp)
        feature_block(tag, "superpoint", tpl, smp, pt, ps, summary)
    except Exception as e:
        summary["SuperPoint"] = {"error": str(e)}
        log(f"  SuperPoint 失败: {e}")
    try:
        pt, ps = loftr_match(tpl, smp)
        feature_block(tag, "loftr", tpl, smp, pt, ps, summary)
    except Exception as e:
        summary["LoFTR"] = {"error": str(e)}
        log(f"  LoFTR 失败: {e}")

    # Farnebäck（对齐对，残余形变场 = |flow|）
    try:
        t0 = time.time()
        flow = cv2.calcOpticalFlowFarneback(tpl, smp_al, None, 0.5, 6, 41, 10, 7, 1.5,
                                            cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        warp_fl = warp_by_field(smp_al, flow[..., 0].astype(np.float32),
                                flow[..., 1].astype(np.float32))
        overlay(tpl, warp_fl, os.path.join(OUT, f"{tag}_ov_flow.png"),
                "Farneback on aligned pair (R=tpl G=sample)")
        st8 = 8
        mag = np.linalg.norm(flow[::st8, ::st8], axis=-1)
        vmax_d = float(max(3.0, np.percentile(mag, 95)))
        heatmap_dense(mag, os.path.join(OUT, f"{tag}_hm_flow.png"),
                      "H 对齐后残余形变场 |flow|（Farneback）", vmax_d)
        summary["flow"] = {"resid_med": float(np.median(mag))}
        log(f"Farnebäck: {time.time() - t0:.1f}s, 残余形变 med {np.median(mag):.2f}px")
    except Exception as e:
        summary["flow"] = {"error": str(e)}
        log(f"  Farnebäck 失败: {e}")

    return summary


def main():
    global BASE, OUT
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if len(sys.argv) > 2:
        BASE = sys.argv[2]
    invert_tpl = len(sys.argv) > 3 and sys.argv[3] == "invert_tpl"
    ds_name = os.path.basename(os.path.normpath(BASE))
    OUT = os.path.join(HERE, f"val_{ds_name}_output" + ("_inv" if invert_tpl else ""))
    os.makedirs(OUT, exist_ok=True)
    log(f"数据集: {BASE} → 输出 {OUT}" + ("（模板极性反转）" if invert_tpl else ""))
    todo = PAIRS if arg == "all" else [(int(arg), int(arg))]
    all_sum = {}
    for ti, si in todo:
        try:
            all_sum[f"p{ti}"] = validate_pair(ti, si, invert_tpl=invert_tpl)
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_sum[f"p{ti}"] = {"error": str(e)}
        with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(all_sum, f, indent=2, ensure_ascii=False)
    log(f"全部完成 → {OUT}/summary.json")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
兜底切换链 fallback_chain.py：先验失效时的分级切换逻辑（可运行实现）。

先验源序列（按成本递增，全部服务于同一条 NCC 主链路精化）：
  1. external   外部先验（机械坐标 / Mark 点粗定位 / 手动对齐，调用方传入 H0）
  2. history    历史先验（同产品上次成功收敛的 H，JSON 存储）
  3. coarse_fm  自动粗匹配：1/4 分辨率下 FM 测 (θ,s,t) → H0（无先验时自动补先验）
  4. aliked     ALIKED+LightGlue 无先验匹配 → H → 回炉 NCC 精化
  5. fm_ecc     FM→ECC 全局链路（纯经典无先验）
  6. fail-safe  全部失败：不出变换，报警人工介入

设计原则（来自本项目实验教训）：
  - 统一验收门 accept()：所有通道共用同一把尺（局部 ZNCC 中位 + 残差），
    不信任任何算法的自评分——评估指标必须与算法目标函数解耦。
  - 兜底只负责"补先验"：ALIKED 的 H 仅作 H0 回炉 NCC pass2/3，最终精度按主链路口径交付。
  - 网格/场构造一律行主序（meshgrid）——本项目两次踩过列主序 reshape 的坑。
  - 超时预算：每级登记耗时上限，超预算跳下一级（产线节拍约束）。

输出：fallback_chain_output/run_summary.json
"""
import cv2
import numpy as np
import os, sys, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import synthetic_tps_validation as stv
from retest_phase_corr import calibrate, fm_full, compose_H
from classical_feature_test import refine, mag_h
from make_overlay_heatmap import local_match

OUT = os.path.join(HERE, "fallback_chain_output")
os.makedirs(OUT, exist_ok=True)
PRIOR_STORE_PATH = os.path.join(OUT, "prior_store.json")

# ---- 验收门参数（实测依据见报告 §11） ----
ACCEPT_GRID_STEP = 128      # 验收网格间距
ACCEPT_MIN_RATIO = 0.30     # 局部验证通过率下限
ACCEPT_RESID_MED = 3.0      # 残差中位上限 px
NCC_MIN_INLIER = 0.30       # 主链路一致性内点率下限（低于此判先验失效）
COARSE_SCALE = 4            # 自动粗匹配降采样倍数
STAGE_BUDGET = {"ncc": 25, "coarse_fm": 10, "aliked": 20, "fm_ecc": 40}  # 秒


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------- 历史先验存取 ----------------

class PriorStore:
    """同产品上次成功收敛的 H（3x3），JSON 落盘。产线上按产品 ID 分键。"""

    def __init__(self, path=PRIOR_STORE_PATH):
        self.path = path
        self._data = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self._data = json.load(f)

    def load(self, key):
        H = self._data.get(key)
        return np.array(H, dtype=np.float64) if H else None

    def save(self, key, H):
        self._data[key] = np.asarray(H, dtype=np.float64).tolist()
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)


# ---------------- NCC 主链路（三遍粗到细，可带 H0 先验） ----------------

def ncc_three_pass(tpl, smp, H0=None):
    """复刻 stv 合成实验的三遍管线：256/80 → 256/48 → 96/32。
    返回 (H 或 None, diag)。一致性内点率过低 = 先验失效信号。"""
    diag = {"pass_inliers": []}
    Hm = H0
    for i, (srch, patch, ncc_min) in enumerate(
            [(stv.SEARCH_R, stv.PATCH, stv.NCC_MIN),
             (48, stv.PATCH, stv.NCC_MIN),
             (stv.SMALL_SEARCH, stv.SMALL_PATCH, stv.SMALL_NCC)]):
        pts, disp, sc = stv.match_grid(tpl, smp, H0=Hm, search_r=srch, patch=patch)
        keep = stv.consistency_filter(pts, disp, sc, ncc_min=ncc_min)
        diag["pass_inliers"].append(int(keep.sum()))
        if keep.sum() < 30:                      # 内点绝对数不足, 无法约束单应
            return None, diag
        if i == 0 and keep.mean() < NCC_MIN_INLIER:
            return None, diag          # 第一遍内点率过低 → 判先验失效
        Hm, _ = stv.fit_homography(pts[keep], disp[keep])
    return Hm, diag


# ---------------- 统一验收门 ----------------

def accept(tpl, smp, Hm, step=ACCEPT_GRID_STEP):
    """所有通道共用：网格映射 + 局部 ZNCC 验证 + 残差统计。行主序网格！"""
    gx = np.arange(step // 2, tpl.shape[1] - step // 2, step, dtype=np.float64)
    gy = np.arange(step // 2, tpl.shape[0] - step // 2, step, dtype=np.float64)
    Xg, Yg = np.meshgrid(gx, gy)                       # 行主序（列主序会搅乱映射）
    grid = np.stack([Xg.ravel(), Yg.ravel()], axis=-1)
    wr = np.c_[grid, np.ones(len(grid))] @ Hm.T
    mapped = wr[:, :2] / wr[:, 2:3]
    found, sc = local_match(tpl, smp, grid, mapped)
    ok = np.isfinite(found[:, 0]) & (sc >= 0.5)
    ratio = float(ok.mean())
    if ratio < ACCEPT_MIN_RATIO:
        return False, {"inlier_ratio": ratio, "reason": "局部验证通过率过低"}
    err = np.linalg.norm(mapped[ok] - found[ok], axis=1)
    med = float(np.median(err))
    m = {"inlier_ratio": ratio, "resid_med": med,
         "resid_p95": float(np.percentile(err, 95)), "zncc_med": float(np.median(sc[ok]))}
    return med <= ACCEPT_RESID_MED and m["zncc_med"] >= 0.6, m


# ---------------- 先验源 3：低分辨率自动粗匹配（FM） ----------------

def coarse_fm_prior(tpl, smp, scale=COARSE_SCALE):
    """1/4 分辨率下跑 FM 自标定 + fm_full → H0。
    自标定与图像内容无关（warp 模板自身），符号约定在任意分辨率下一致。
    先验只需 ~50px 精度，1/4 分辨率的 FM 精度（~1px@1/4x = 4px@全分辨率）足够。"""
    h, w = tpl.shape
    w_lr, h_lr = w // scale, h // scale
    t_lr = cv2.resize(tpl, (w_lr, h_lr), interpolation=cv2.INTER_AREA)
    s_lr = cv2.resize(smp, (w_lr, h_lr), interpolation=cv2.INTER_AREA)
    cal = calibrate(t_lr, w_lr, h_lr, "coarse")
    if not cal[0]:
        return None, {"reason": "粗分辨率自标定失败（极相关找错峰）"}
    H_lr, *_ = fm_full(t_lr, s_lr, cal, w_lr, h_lr)
    # 低分辨率 H → 工作分辨率：q_w = s·q_lr, p_lr = p_w/s → H_w = D·H_lr·D⁻¹
    D = np.diag([1.0 * scale, 1.0 * scale, 1.0])
    Di = np.diag([1.0 / scale, 1.0 / scale, 1.0])
    return D @ H_lr @ Di, {}


# ---------------- 先验源 4：ALIKED+LightGlue（无先验） ----------------

def stage_aliked(tpl, smp):
    try:
        import torch
        from lightglue import LightGlue
        from lightglue.aliked import ALIKED
    except ImportError as e:
        return None, {"reason": f"torch/lightglue 不可用: {e}"}
    ext = ALIKED(max_num_keypoints=8000).eval()
    mat = LightGlue(features="aliked").eval()
    tt = lambda im: torch.from_numpy(im.astype(np.float32) / 255.0)[None, None]
    with torch.inference_mode():
        f_t, f_s = ext.extract(tt(tpl)), ext.extract(tt(smp))
        pred_ = mat({"image0": f_t, "image1": f_s})
        m = pred_["matches"][0].cpu().numpy()
        pt = f_t["keypoints"][0].cpu().numpy().astype(np.float64)[m[:, 0]]
        ps = f_s["keypoints"][0].cpu().numpy().astype(np.float64)[m[:, 1]]
    H_raw, inl = mag_h(pt, ps)
    pr, nr = refine(tpl, smp, pt, ps)
    okr = nr >= 0.5
    H_ref, inl_r = mag_h(pt[okr], pr[okr])
    return H_ref, {"lg_matches": int(len(m)), "raw_inliers": int(inl.sum()),
                   "refined_inliers": int(inl_r.sum())}


# ---------------- 先验源 5：FM→ECC 全局链路 ----------------

def stage_fm_ecc(tpl, smp):
    Wf, Hf = tpl.shape[1], tpl.shape[0]
    cal = calibrate(tpl, Wf, Hf, "PCB")
    if not cal[0]:
        return None, {"reason": "全分辨率自标定失败"}
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
    return Hm, {}


# ---------------- 主切换状态机 ----------------

def run(tpl, smp, external_H0=None, product_key="default", budget_s=90, trace=None):
    """返回 (H 或 None, 命中通道, metrics, trace)。成功即写入历史先验。"""
    trace = trace if trace is not None else []
    store = PriorStore()
    t_start = time.time()

    def remaining():
        return budget_s - (time.time() - t_start)

    # ---- 先验源序列：external → history → coarse_fm（惰性：历史命中则不花粗匹配的 4s）----
    coarse_holder = {}

    def prior_external():
        return external_H0

    def prior_history():
        return store.load(product_key)

    def prior_coarse():
        if "H" not in coarse_holder:
            t0 = time.time()
            H, diag_c = coarse_fm_prior(tpl, smp)
            coarse_holder["H"] = H
            trace.append({"stage": "coarse_fm_prior", "s": round(time.time() - t0, 1),
                          **diag_c, "ok": H is not None})
            log(f"自动粗匹配: {'成功' if H is not None else '失败'} ({time.time() - t0:.1f}s)")
        return coarse_holder["H"]

    priors = []
    if external_H0 is not None:
        priors.append(("external", prior_external))
    if store.load(product_key) is not None:
        priors.append(("history", prior_history))
    priors.append(("coarse_fm", prior_coarse))

    for src, maker in priors:
        if remaining() < STAGE_BUDGET["ncc"]:
            trace.append({"stage": f"ncc({src})", "reason": "预算不足, 跳过"})
            continue
        H0 = maker()
        if H0 is None:
            trace.append({"stage": f"ncc({src})", "reason": "先验源不可用"})
            continue
        t0 = time.time()
        Hm, diag = ncc_three_pass(tpl, smp, H0=H0)
        ok, m = (False, {"reason": "pass1 内点率过低 → 先验失效"}) if Hm is None else accept(tpl, smp, Hm)
        trace.append({"stage": f"ncc({src})", "s": round(time.time() - t0, 1),
                      "pass_inliers": diag.get("pass_inliers"), **m, "ok": bool(ok)})
        log(f"NCC 主链路 [先验={src}]: {'验收通过' if ok else '验收失败'} ({time.time() - t0:.1f}s) {m}")
        if ok:
            store.save(product_key, Hm)
            return Hm, f"ncc+prior:{src}", m, trace

    # ---- 兜底 4：ALIKED+LG → 回炉 NCC 精化 ----
    if remaining() >= STAGE_BUDGET["aliked"]:
        t0 = time.time()
        H_al, diag_a = stage_aliked(tpl, smp)
        if H_al is not None:
            Hm, diag = ncc_three_pass(tpl, smp, H0=H_al)
            ok, m = (False, {}) if Hm is None else accept(tpl, smp, Hm)
            trace.append({"stage": "aliked+ncc_refine", "s": round(time.time() - t0, 1),
                          **diag_a, **m, "ok": bool(ok)})
            log(f"ALIKED→NCC 回炉: {'验收通过' if ok else '验收失败'} ({time.time() - t0:.1f}s)")
            if ok:
                store.save(product_key, Hm)
                return Hm, "aliked+ncc", m, trace
        else:
            trace.append({"stage": "aliked", "s": round(time.time() - t0, 1), **diag_a, "ok": False})
            log(f"ALIKED 不可用: {diag_a.get('reason')}")

    # ---- 兜底 5：FM→ECC 全局链路 ----
    if remaining() >= STAGE_BUDGET["fm_ecc"]:
        t0 = time.time()
        H_ecc, diag_e = stage_fm_ecc(tpl, smp)
        if H_ecc is not None:
            ok, m = accept(tpl, smp, H_ecc)
            trace.append({"stage": "fm_ecc", "s": round(time.time() - t0, 1), **m, "ok": bool(ok)})
            log(f"FM→ECC: {'验收通过' if ok else '验收失败'} ({time.time() - t0:.1f}s)")
            if ok:
                store.save(product_key, H_ecc)
                return H_ecc, "fm_ecc", m, trace
        else:
            trace.append({"stage": "fm_ecc", "s": round(time.time() - t0, 1), **diag_e, "ok": False})

    # ---- fail-safe：不出变换 ----
    trace.append({"stage": "fail_safe", "ok": False})
    log("全部通道失败 → FAIL-SAFE：不出变换，报警人工介入")
    return None, "fail_safe", {}, trace


# ---------------- 演练：正常 / 破坏先验 / 历史命中 ----------------

def main():
    TPL_PATH = r"D:\Users\Administrator\Desktop\weights222\object_region_full_res.jpg"
    SMP_PATH = r"D:\Users\Administrator\Desktop\weights222\corresponding_sample_region_full_res.jpg"
    DS = 0.5
    tpl0 = cv2.imread(TPL_PATH, cv2.IMREAD_GRAYSCALE)
    smp0 = cv2.imread(SMP_PATH, cv2.IMREAD_GRAYSCALE)
    Wf, Hf = int(tpl0.shape[1] * DS), int(tpl0.shape[0] * DS)
    tpl = cv2.resize(tpl0, (Wf, Hf), interpolation=cv2.INTER_AREA)
    smp = cv2.resize(smp0, (Wf, Hf), interpolation=cv2.INTER_AREA)

    if os.path.exists(PRIOR_STORE_PATH):
        os.remove(PRIOR_STORE_PATH)      # 演练从干净状态开始

    # 场景 A：外部先验被破坏（错 300px）→ 应逐级降级到 coarse_fm → NCC 成功
    log("== 场景 A: 外部先验失效 (错 300px) ==")
    bad_H0 = compose_H(0.0, 1.0, 300.0, -150.0, Wf, Hf)
    t0 = time.time()
    H, hit, m, tr = run(tpl, smp, external_H0=bad_H0, product_key="weights222_demo", budget_s=120)
    log(f"场景 A 结果: 通道={hit}, {time.time() - t0:.1f}s, {m}\n")

    # 场景 B：再次运行 → 历史先验命中，快速路径
    log("== 场景 B: 历史先验命中 ==")
    t0 = time.time()
    H2, hit2, m2, tr2 = run(tpl, smp, product_key="weights222_demo", budget_s=60)
    log(f"场景 B 结果: 通道={hit2}, {time.time() - t0:.1f}s, {m2}\n")

    with open(os.path.join(OUT, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"scenario_A": tr, "scenario_B": tr2}, f, indent=2, ensure_ascii=False)
    log(f"完成 → fallback_chain_output/run_summary.json")


if __name__ == "__main__":
    main()

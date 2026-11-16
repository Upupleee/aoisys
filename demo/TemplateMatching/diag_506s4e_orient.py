# -*- coding: utf-8 -*-
"""506s4e 几何变换穷举：水平/垂直/180° 翻转 × 极性，找先验内点率跳变的那一个。"""
import cv2
import numpy as np
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from val_2652s2g import load_down, coarse_prior, accept

tpl_raw = load_down(r"D:\AOI\aoisys\506s4e\template\template_1.tif", 8)
smp_raw = load_down(r"D:\AOI\aoisys\506s4e\sample\test_imgs1.tif", 8)
tpl = 255 - tpl_raw                      # 极性对齐（固定：模板取反）

variants = {
    "smp 原图": smp_raw,
    "smp 水平翻转(flip1)": cv2.flip(smp_raw, 1),
    "smp 垂直翻转(flip0)": cv2.flip(smp_raw, 0),
    "smp 180°(flip-1)": cv2.flip(smp_raw, -1),
    "smp 转置(T)": cv2.transpose(smp_raw),
}

print(f"模板(取反) mean={tpl.mean():.0f} std={tpl.std():.0f}")
for name, smp in variants.items():
    try:
        H0, diag = coarse_prior(tpl, smp)
        ok0, m0, *_ = accept(tpl, smp, H0)
        ir = m0.get("inlier_ratio", 0)
        rm = m0.get("resid_med", -1)
        print(f"{name:22s}: scale={diag.get('scale'):.3f} θ={diag.get('dtheta'):+.3f}° "
              f"zncc={diag.get('zncc'):.3f} → 先验内点率 {ir:.1%} 残差med {rm:.1f}px "
              f"{'★' if ir > 0.3 else ''}")
    except Exception as e:
        print(f"{name:22s}: 失败 {e}")

# -*- coding: utf-8 -*-
"""三批数据模板/采样灰度统计与形态对比：同是图纸转换，为何 2652 能配 506 不能？"""
import cv2
import numpy as np
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from val_2652s2g import load_down

SETS = [
    ("w222", r"D:\Users\Administrator\Desktop\weights222\object_region_full_res.jpg",
     r"D:\Users\Administrator\Desktop\weights222\corresponding_sample_region_full_res.jpg", 2),
    ("2652", r"D:\AOI\aoisys\2652s2g-gtl\template\template_1.tif",
     r"D:\AOI\aoisys\2652s2g-gtl\sample\test_imgs1.tif", 8),
    ("506", r"D:\AOI\aoisys\506s4e\template\template_1.tif",
     r"D:\AOI\aoisys\506s4e\sample\test_imgs1.tif", 8),
]

for name, tp, sp, ds in SETS:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    def load_gray(path, scale):
        im = Image.open(path)
        im.load()
        arr = np.asarray(im)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        h, w = arr.shape
        return cv2.resize(np.ascontiguousarray(arr), (w // scale, h // scale),
                          interpolation=cv2.INTER_AREA)
    tpl = load_gray(tp, ds)
    smp = load_gray(sp, ds)
    print(f"== {name} ==")
    print(f"  模板: mean={tpl.mean():.1f} std={tpl.std():.1f} "
          f"p5={np.percentile(tpl,5):.0f} p50={np.percentile(tpl,50):.0f} p95={np.percentile(tpl,95):.0f}")
    print(f"  采样: mean={smp.mean():.1f} std={smp.std():.1f} "
          f"p5={np.percentile(smp,5):.0f} p50={np.percentile(smp,50):.0f} p95={np.percentile(smp,95):.0f}")
    # 中心区域拼图（等高裁剪，均衡前后各一行）
    h = min(tpl.shape[0], smp.shape[0])
    w = min(tpl.shape[1], smp.shape[1])
    ty, tx = (tpl.shape[0] - h) // 2, (tpl.shape[1] - w) // 2
    sy, sx = (smp.shape[0] - h) // 2, (smp.shape[1] - w) // 2
    tc, sc = tpl[ty:ty + h, tx:tx + w], smp[sy:sy + h, sx:sx + w]
    eq_t = cv2.equalizeHist(tc)
    eq_s = cv2.equalizeHist(sc)
    half = (w // 4, h // 4)
    row = np.hstack([cv2.resize(tc, half), cv2.resize(sc, half),
                     cv2.resize(eq_t, half), cv2.resize(eq_s, half)])
    out = os.path.join(HERE, f"diag_domain_{name}.png")
    cv2.imwrite(out, row)
    print(f"  -> {out}  (左→右: 模板原图 | 采样原图 | 模板均衡 | 采样均衡)")

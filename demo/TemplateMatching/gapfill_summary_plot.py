# -*- coding: utf-8 -*-
"""缺口补测汇总对比图（数值取自各 gapfill 实验 results.json / summary）。"""
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "gapfill_summary")
os.makedirs(OUT, exist_ok=True)

# 特征匹配类（@0.5x，同口径：raw / NCC精化后）
feat = [
    # name, raw_n, raw_ratio, raw_med, ref_n, ref_ratio, ref_med, ref_rms
    ("NCC pass1 (base)", 407, 0.983, 0.50, 407, 0.983, 0.50, 0.66),
    ("ALIKED+LG", 4599, 0.801, 2.36, 4050, 0.944, 0.74, 1.23),
    ("LoFTR", 463, 0.883, 1.87, 316, 0.687, 0.96, 1.47),
    ("SP+LG", 3665, 0.316, 4.10, 2193, 0.808, 1.35, 1.93),
    ("SIFT", 426, 0.054, 0.23, 21, 0.053, 0.31, 0.39),
    ("ORB", 1583, 0.073, 0.36, 112, 0.071, 0.59, 0.80),
]
names = [f[0] for f in feat]
x = np.arange(len(feat))

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
ax[0].bar(x - 0.2, [f[2] * 100 for f in feat], 0.4, label="raw inlier ratio")
ax[0].bar(x + 0.2, [f[5] * 100 for f in feat], 0.4, label="refined inlier ratio")
ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=25, fontsize=8)
ax[0].set_ylabel("MAGSAC inlier ratio (%)"); ax[0].legend(); ax[0].grid(alpha=0.3, axis="y")
ax[0].set_title("Inlier ratio")
ax[1].bar(x - 0.2, [f[3] for f in feat], 0.4, label="raw med (px)")
ax[1].bar(x + 0.2, [f[6] for f in feat], 0.4, label="refined med (px)")
ax[1].set_xticks(x); ax[1].set_xticklabels(names, rotation=25, fontsize=8)
ax[1].set_ylabel("inlier resid median (px @0.5x)"); ax[1].legend(); ax[1].grid(alpha=0.3, axis="y")
ax[1].set_title("Residual accuracy")
fig.suptitle("Feature matching methods (real pair weights222)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "feature_methods_compare.png"), dpi=130)
plt.close(fig)

# 全局粗定位器：H 级稠密偏差（log 轴）
init = [
    ("NCC pass1 (base)", 0.66),
    ("ECC pyramid", 23.61),
    ("PhaseCorr+logpolar", 53.89),
]
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.bar([i[0] for i in init], [i[1] for i in init], color=["#4a7dbd", "#c0504d", "#9bbb59"])
for i, (n, v) in enumerate(init):
    ax.text(i, v * 1.05, f"{v:.2f}px", ha="center", fontsize=9)
ax.set_yscale("log"); ax.set_ylabel("dense H dev median (px @0.5x)")
ax.set_title("Global coarse initializers (lower is better)")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "global_init_compare.png"), dpi=130)
plt.close(fig)

lines = [
    "缺口补测汇总（真实对 weights222 @0.5x）",
    "",
    "== 特征匹配类 ==",
    f"{'方法':<16}{'匹配':>6}{'raw内点率':>10}{'raw_med':>9}{'精化内点率':>11}{'精化med':>9}{'精化RMS':>9}",
]
for f in feat:
    lines.append(f"{f[0]:<16}{f[1]:>6}{f[2]:>10.1%}{f[3]:>9.2f}{f[5]:>11.1%}{f[6]:>9.2f}{f[7]:>9.2f}")
lines += [
    "",
    "== 全局粗定位器 ==",
    "PhaseCorr: 平移接近(差~2px)但log-polar角度-0.74°(真值+0.23°)，H偏差med 53.9px —— 符号/通道约定需调试",
    "ECC金字塔: 稳定收敛(自举/相位初值同结果)但锁在偏差23.6px的局部最优，cc=0.50 —— 适合做精化器不适合做初值",
    "",
    "== 稠密形变 ==",
    "Farnebäck: 27.7s, 流场vs H med=1.11px (P95 36.2, 长尾来自弱纹理区); vs 控制点 med=0.79px —— 加置信度掩码后可作为TPS的替代/补充",
    "",
    "== 学习型补充 ==",
    "SuperPoint+LG: raw 31.6% (ALIKED为80.1%), 精化后 80.8%/1.35px —— 泛化不如ALIKED",
    "LoFTR outdoor: raw 88.3%(特征类最高)但仅463对, 精化后68.7%/0.96px —— 鲁棒性最强、覆盖最低",
]
with open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))

# -*- coding: utf-8 -*-
"""
9 类算法横向独立对比拼图：
  1. w222_gallery.png            weights222 九类最终配准贴图 3x3
  2. p{1..4}_gallery.png         2652s2g 每对九类最终配准贴图 3x3（失败类占位注明）
  3. compare_all_overview.png    两批数据 5 张九宫格縮略总对比

独立口径说明（写入每张图脚注）：
  - SIFT/ORB/ALIKED+LG/SP+LG/LoFTR：原始对独立运行（特征法天然抗缩放，无先验）
  - FM→ECC：FM 粗测(θ,s,t) 自己完成全局粗配准后 ECC 精化（无先验、独立）
  - NCC-H / NCC+TPS / Farnebäck：2652 存在 8.5~10.8% 缩放差，
    这三种"平移级"方法以"像素比例已对齐"为前提，在 FM→ECC 矫正对上独立精化
    （weights222 缩放≈1，可直接在原始对独立运行）。
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

HERE = os.path.dirname(os.path.abspath(__file__))
W222 = os.path.join(HERE, "overlay_heatmap_output")
V2652 = os.path.join(HERE, "val_2652s2g_output")

NOTE = ("独立口径：SIFT/ORB/ALIKED+LG/SP+LG/LoFTR 原始对独立运行；FM→ECC 自带粗配准（无先验）；"
        "NCC-H/NCC+TPS/Farnebäck 为平移级精化——2652 存在 8.5~10.8% 缩放差，"
        "在 FM→ECC 矫正对上独立精化（w222 缩放≈1 直接原始对独立运行）。红=模板 绿=配准采样 黄=对齐")

W222_ITEMS = [
    ("NCC-H  med 0.36px", "ov_ncc_h.png"),
    ("NCC+TPS  留出med 0.21px", "ov_ncc_tps.png"),
    ("FM→ECC  med 0.59px", "ov_fm_ecc.png"),
    ("SIFT  内点21 H退化", "ov_sift.png"),
    ("ORB  med 0.59px (112)", "ov_orb.png"),
    ("ALIKED+LG  med 0.74px (4050)", "ov_aliked.png"),
    ("SP+LG  med 1.35px (2193)", "ov_superpoint.png"),
    ("LoFTR  med 0.96px (316)", "ov_loftr.png"),
    ("Farnebäck  残余med 1.11px", "ov_flow.png"),
]

# 2652 每对的 med（来自 val_2652s2g_output/summary.json 与日志）
P2652 = {
    1: {"ncc_h": "0.91px", "ncc_tps": "留出med 1.76px", "fm_ecc": "0.86px",
        "sift": "0.36px (38)", "orb": "0.43px (25)", "aliked": "1.74px (397)",
        "superpoint": "1.86px (602)", "loftr": "失败 (0 对)", "flow": "残余 0.26px"},
    2: {"ncc_h": "0.16px", "ncc_tps": "留出med 5.13px", "fm_ecc": "0.53px",
        "sift": "0.98px (158)", "orb": "1.00px (108)", "aliked": "1.69px (488)",
        "superpoint": "1.60px (683)", "loftr": "失败 (1 对)", "flow": "残余 0.31px"},
    3: {"ncc_h": "0.10px", "ncc_tps": "留出med 5.35px", "fm_ecc": "0.52px",
        "sift": "0.75px (114)", "orb": "0.70px (29)", "aliked": "1.48px (109)",
        "superpoint": "1.79px (710)", "loftr": "4 对", "flow": "残余 0.31px"},
    4: {"ncc_h": "0.67px", "ncc_tps": "门控禁用", "fm_ecc": "0.60px",
        "sift": "0.90px (126)", "orb": "0.45px (24)", "aliked": "1.67px (157)",
        "superpoint": "1.64px (286)", "loftr": "6 对", "flow": "残余 0.00px"},
}

NAMES = [("NCC-H", "ov_ncc_h.png", "ncc_h"),
         ("NCC+TPS", "ov_ncc_tps.png", "ncc_tps"),
         ("FM→ECC", "ov_fm_ecc.png", "fm_ecc"),
         ("SIFT", "ov_sift.png", "sift"),
         ("ORB", "ov_orb.png", "orb"),
         ("ALIKED+LG", "ov_aliked.png", "aliked"),
         ("SP+LG", "ov_superpoint.png", "superpoint"),
         ("LoFTR", "ov_loftr.png", "loftr"),
         ("Farnebäck", "ov_flow.png", "flow")]


def grid9(items, title, out, figsize=(14, 24)):
    """items: [(标题, 文件路径或None)] 3x3。"""
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    axes = np.atleast_2d(axes)
    for k in range(9):
        ax = axes[k // 3, k % 3]
        ttl, path = items[k]
        if path and os.path.exists(path):
            ax.imshow(mpimg.imread(path))
            ax.set_title(ttl, fontsize=11)
        else:
            ax.text(0.5, 0.5, "失败\n(无有效匹配)", ha="center", va="center",
                    fontsize=16, color="r", transform=ax.transAxes)
            ax.set_title(ttl, fontsize=11)
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.text(0.5, 0.005, NOTE, ha="center", fontsize=7, wrap=True)
    fig.tight_layout(rect=[0, 0.012, 1, 0.97])
    fig.savefig(out, dpi=90)
    plt.close(fig)
    print(f"-> {out}")


def main():
    import numpy as np

    # 1. weights222 九宫格
    items = [(t, os.path.join(W222, f)) for t, f in W222_ITEMS]
    grid9(items, "weights222：9 类算法最终配准贴图（同分辨率、重复纹理网格板）",
          os.path.join(HERE, "match_gallery_output", "w222_gallery.png"))

    # 2. 2652 每对九宫格
    with open(os.path.join(V2652, "summary.json"), "r", encoding="utf-8") as f:
        summ = json.load(f)
    for i in (1, 2, 3, 4):
        med = P2652[i]
        items = []
        for disp, f, key in NAMES:
            p = os.path.join(V2652, f"p{i}_{f}")
            items.append((f"{disp}  {med[key]}", p if os.path.exists(p) else None))
        grid9(items, f"2652s2g pair{i}：9 类算法最终配准贴图（缩放差样本，单面字符板）",
              os.path.join(V2652, f"p{i}_gallery.png"))

    # 3. 总对比：w222 + 4 对缩略
    grids = [("weights222 (缩放≈1, 重复纹理网格板)", os.path.join(HERE, "match_gallery_output", "w222_gallery.png"))]
    grids += [(f"2652 pair{i} (缩放差~{8.5+i*0.75:.1f}%, 单面字符板)",
               os.path.join(V2652, f"p{i}_gallery.png")) for i in (1, 2, 3, 4)]
    fig, axes = plt.subplots(len(grids), 1, figsize=(10, 14 * len(grids)))
    axes = np.atleast_1d(axes)
    for ax, (ttl, p) in zip(axes, grids):
        ax.imshow(mpimg.imread(p))
        ax.set_title(ttl, fontsize=13)
        ax.axis("off")
    fig.savefig(os.path.join(V2652, "compare_all_overview.png"), dpi=70)
    plt.close(fig)
    print(f"-> {os.path.join(V2652, 'compare_all_overview.png')}")


if __name__ == "__main__":
    main()

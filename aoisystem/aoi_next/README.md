# AOI Next：离线链路渐进迁移

本目录是独立新版的第一阶段，不替换 `LoginWindow.py`，不修改旧窗口、硬件或算法实现。当前迁移粒度为一对明确指定的模板/样本条带，不会自动发现、排序或拼接整板条带。

## 边界

- `core.py`：不可变任务、缺陷、结果，检测与仓储抽象，成功/失败编排。仅 Python 标准库，无 Qt、图像或模型依赖。
- `pipeline.py`：配准接口、图像检测接口、流水线及坐标转换。领域 bbox 为 `xyxy`。
- `adapters.py`：延迟加载旧 ALIKED + LightGlue 配准和仓库定制 SAHI；独立实例模型缓存，无旧 ModelManager 单例调用。
- `repository.py`：SQLite 事务保存任务参数、状态和缺陷；重复运行生成新记录，禁止覆盖 run_id。
- `__main__.py`：独立命令行入口与依赖装配。业务失败返回 1，启动/保存失败返回 2。

依赖方向为：命令行装配 → 流水线与适配器 → 业务接口。业务核心不导入适配器或 UI。配准适配器仍调用旧算法模块；旧模块内部仍导入旧目录的 ModelManager 类型及其他算法依赖，这是明确保留的迁移边界，不代表算法内部已全部解耦。

## 使用

在仓库根目录、已有算法依赖的 Python 3.10+ 环境运行：

```powershell
python -B -m aoi_next --help
python -B -m aoi_next --task-id board001-A-strip01 --sample D:/data/sample01.tif --template D:/data/template01.tif --weights D:/models/detection.pt --database D:/data/aoi-next.sqlite3 --device cuda:0 --crop-top 9070
python -B -m unittest discover -s tests -p "test_*.py" -v
```

父目录必须已存在。路径也支持 Linux。当前示例不代表 Windows 硬件或 CUDA 环境已经适配。不要直接使用根目录混合了 Conda 系统包的 requirements.txt 创建新环境。

真实推理沿用项目已有的 NumPy、Pillow、Torch、OpenCV、Kornia、SciPy、Matplotlib、Ultralytics、LightGlue 及本地 SAHI 依赖。本阶段未提供经干净环境验收的锁文件。ALIKED/LightGlue 权重需提前缓存；首次使用旧算法可能联网下载，不保证断网首启。

## 与旧链路的差异

- 默认保留样本顶部裁剪 9070 行、网格 7、每格采样 100、CLAHE 和 1200 切片；这些只是兼容默认值，不代表已证明正确的现场标定参数。
- `--crop-top 0` 适用于已经预裁剪输入。末条带需显式指定 `--sample-width-ratio`，保留样本右侧相应比例，不再通过隐含文件编号猜测。
- 检测仍按上下半图执行，下半图偏移使用实际上半图高度，仅平移 y，不改变框高度；奇数高度和单行图像有单独测试。
- 新检测适配器把 confidence 显式传给 YOLO 批量推理，不沿用旧代码漏传阈值的行为。因此真实结果可能变化，需固定样本对照验收。
- 仅支持轴对齐目标检测权重。暂不保证 TensorRT engine 的动态批量和设备兼容性，优先在现场既有环境验证 `.pt`。
- 所有坐标都是配准叠加图像素坐标，不是原始相机坐标、整板坐标或设备毫米坐标。旧配准可能缩放到最长边 30000；本阶段不提供逆变换或标定补偿。
- 不修改旧 JSON 数据，不写回旧扫描记录。新版业务数据仅写显式指定的 SQLite 数据库。
- 暂不输出 COCO/VRS 文件、不生成可视化图、不控制硬件；旧 VRS 契约存在歧义，不能盲目迁移。类别标签暂存类别 ID 字符串。
- 核心支持注入替身测试；真实模型适配器按串行实例使用，不允许多个线程共享同一模型实例执行推理。

## 验收与后续迁移

标准库回归覆盖任务不可变、成功/失败隔离、SQLite 往返/事务、非法输出、坐标平移、上下半图和命令行失败路径。替身测试不证明真实配准精度、召回率、速度或 GPU 可用。

下一步先在原 Linux 算法环境准备固定样本、权重和预期输出，验证完整性、坐标、误报及阈值透传；再处理配准质量门控、模型依赖清理、整板条带编排和 COCO/VRS 标定输出。通过后再迁移硬件接口、扫描状态机及 MVVM 界面。旧系统始终保留为对照入口。

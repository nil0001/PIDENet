# PIDENet 抓取位姿离线标注算法复现

---

## 1. 目录结构

```
pidenet_annotator/
├── pidenet_annotator/           # 核心 Python 包
│   ├── __init__.py
│   ├── stable_pose.py           # 3.2.1  凸包 + 稳定姿态分析
│   ├── projection.py            # 3.2.2  正交光线投射 + 动态高度阈值
│   ├── grasp2d.py               # 3.2.3  EFD 拟合 + 双分支抓取点提取
│   ├── approach_vector.py       # 3.2.4  KNN + PCA + 反投影
│   ├── scoring.py               # 3.2.5  四项打分 + 摩擦锥可行性
│   ├── pipeline.py              # 主流程编排
│   └── viz3d.py                 # Plotly 3D 可视化辅助
├── run_phase1.py                # 主运行脚本
├── make_debug_figure.py         # 生成论文风格全流程调试图
├── hyperparams.yml              # 所有超参数（含注释）
├── requirements.txt             # pip 依赖清单
├── pyproject.toml               # 可选：pip install -e . 时用
└── README.md                    # 本文件
```

---

## 2. 环境配置

### 2.1 Windows

**推荐使用 Python 3.10 / 3.11 / 3.12**（3.13 目前部分依赖轮子还没跟上）。

**第一步：安装 Python**（如已有可跳过）
- 从 <https://www.python.org/downloads/windows/> 下载安装包，安装时**务必勾选** *"Add Python to PATH"*。
- 在 `cmd` 或 `PowerShell` 里执行 `python --version` 应能看到 `Python 3.1x.x`。

**第二步：创建虚拟环境**（强烈推荐，避免污染全局环境）

打开 `cmd` 或 `PowerShell`，切换到本项目所在目录（假设你把项目放在 `E:\paper\PIDENet\pidenet_annotator`）：

```bat
cd /d E:\paper\PIDENet\pidenet_annotator
python -m venv .venv
.venv\Scripts\activate
```

激活后命令行前面会出现 `(.venv)` 字样。

**第三步：安装依赖**

```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

正常情况下 5 分钟以内装完（其中 `opencv-python`、`shapely`、`trimesh` 略大）。

**关于 `embreex` 加速**：这是一个可选依赖（把 trimesh 的光线投射提速 10-50 倍）。requirements.txt 里已经标记为 Windows/Linux x86_64 自动安装。如果安装失败也不影响使用，代码会自动 fallback 到纯 Python 实现，只是每个姿态的光线投射会从 0.15 秒变成 3 秒左右——总运行时间从 5 秒变成 30 秒，可以接受。

**第四步：验证安装**

```bat
python -c "import pidenet_annotator; print('OK', pidenet_annotator.__version__)"
```

看到 `OK 0.1.0` 即配置成功。

---

### 2.2 Ubuntu

```bash
# 假设你把项目放在 ~/PIDENet/pidenet_annotator
cd ~/PIDENet/pidenet_annotator
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

python -c "import pidenet_annotator; print('OK', pidenet_annotator.__version__)"
```

Ubuntu 上没有其他额外系统依赖（trimesh / opencv-python 都是自带二进制轮子）。

---

## 3. 运行

### 3.1 你的 Windows 数据集路径

你说明数据集在 `E:\paper\PIDENet\LINEMOD`，目录结构应当是：

```
E:\paper\PIDENet\LINEMOD\
├── models\
│   ├── obj_01.ply     ← 猩猩（ape）
│   ├── obj_02.ply
│   ├── ...
│   └── obj_05.ply     ← 水壶（kettle）
├── data\
│   ├── 01\ ...
│   └── 05\ ...
└── segnet_results\
```

### 3.2 生成候选抓取位姿（主任务）

```bat
python run_phase1.py --linemod E:\paper\PIDENet\LINEMOD --objects 1 5 --out outputs
```

参数说明：
- `--linemod` 指向 Linemod 根目录（脚本会自动去 `models/` 下找 `obj_01.ply` 和 `obj_05.ply`）
- `--objects 1 5` 指定要处理的物体 ID（可以多个空格分隔）
- `--out` 输出目录（不存在会自动创建）
- `--top-n` 可选，控制可视化显示前 N 个候选，默认 8

对每个物体，会在 `outputs/` 生成 3 个文件：

| 文件 | 用途 |
|---|---|
| `phase1_candidates_<tag>.yml` | **核心产出**：物体坐标系下的所有候选，Phase 2 会读取这个 |
| `preview_final_<tag>.png` | 4 视图静态预览（3 个 iso + 1 个俯视），点线颜色对应候选编号 |
| `candidates_3d_<tag>.html` | **交互式** Plotly 3D 可视化，浏览器打开即可拖拽旋转 |

`<tag>` 目前对物体 1 显示为 `ape`、对物体 5 显示为 `kettle`；其他物体显示为 `obj_XX`。

### 3.3 生成论文风格的调试图（可选，用来核对中间步骤）

```bat
python make_debug_figure.py --linemod E:\paper\PIDENet\LINEMOD --objects 1 5 --out outputs
```

会生成 `pipeline_stages_<tag>.png`：每一行是一个稳定姿态，每一列依次是「(a) 正交投影朴素 mask → (b) 上层 30% 高度阈值 mask → (c) 挖除封闭孔洞后的最终 mask → (d) EFD 拟合轮廓 + 双分支抓取点」。

### 3.4 Phase 2 — 生成每帧相机坐标系下的抓取标签

Phase 1 得到的候选是**物体自身坐标系**下的（这些是不变量，物体本身的几何决定）。Phase 2 用 `gt.yml` 里每帧的相机-物体外参把候选变换到**相机坐标系**，并加上一个**帧相关的碰撞感知分数** `P_coll`，最终的每帧分数为 `S = Q_offline × P_coll`（对应论文 Eq.13）。

碰撞检测两部分：
- **自遮挡**：把 CAD 模型按 R,t 摆到相机坐标系里，从相机原点朝每个候选的 pkm（抓取中心，位于夹爪中平面上）发光线。如果最近的表面比 pkm 明显更靠前，说明物体自己的其它部分挡住了这个抓取区，`P_coll` 打折。（论文式的射线是 pkm 而非 p1、p2，是因为 pkm 位于夹爪的空隙里，是"能否从相机方向到达抓取"的直接量化。查 p1、p2 会误伤所有 hole 类型抓取——它们的接触点在夹爪要包裹的壁的两侧。）
- **场景遮挡**：把 pkm、p1、p2 通过内参 K 投影到像素，如果落在 mask 之外（说明这一帧里该区域被其它物体挡了或跑出了画面），也打折。

跑法：

```bat
python run_phase2.py --linemod E:\paper\PIDENet\LINEMOD --phase1 outputs --out outputs --objects 1 5
```

参数：
- `--phase1` 指向 Phase 1 的输出目录（脚本会去里面找 `phase1_candidates_ape.yml` 和 `phase1_candidates_kettle.yml`）
- `--max-frames N` 可选，只处理前 N 帧（用来快速调试）
- `--sample-frames K` 每个物体从数据集中随机采样 K 帧生成对照图（默认 6）
- `--top-n-viz N` 对照图上每帧最多画多少个候选（默认 5）
- `--min-S X` 对照图**仅显示** S≥X 的候选（yml 里始终包含全部）

对每个物体，会在 `outputs/` 生成两个文件：

| 文件 | 用途 |
|---|---|
| `phase2_labels_<tag>.yml` | 每帧的抓取标签，格式为 `frame_id: {pose1:{w,u,v,center,S,...}, pose2:{...}, ...}` |
| `phase2_sample_grid_<tag>.png` | 随机采样 6 帧的 RGB 图，叠加投影出来的抓取候选（S 排名颜色编码） |

**性能参考**：完整数据集（ape 1236 帧 + kettle 1196 帧，共 2432 帧）在装了 embreex 加速的机器上大约 2 分钟跑完。没装 embreex 会慢约 5-10 倍。

### 3.5 完整流程一次性跑

```bat
REM 生成物体坐标系候选
python run_phase1.py --linemod E:\paper\PIDENet\LINEMOD --objects 1 5 --out outputs
REM 可选：生成论文风格调试图
python make_debug_figure.py --linemod E:\paper\PIDENet\LINEMOD --objects 1 5 --out outputs
REM 生成每帧相机坐标系标签
python run_phase2.py --linemod E:\paper\PIDENet\LINEMOD --phase1 outputs --out outputs --objects 1 5
```

也可以双击 `run_windows.bat`（默认已经写好 Phase 1 的调用；如需自动跑 Phase 2 也可以在里面加一行 `python run_phase2.py ...`）。

### 3.6 另一种运行方式：直接指定 PLY 文件（仅 Phase 1）

如果你只想跑 Phase 1、不想按数据集目录来，可以直接指定 ply：

```bat
python run_phase1.py --plys E:\paper\PIDENet\LINEMOD\models\obj_01.ply E:\paper\PIDENet\LINEMOD\models\obj_05.ply --out outputs
```

---

## 4. 快速验证 —— 你应该看到什么

正常运行结束后终端会打印类似下面的内容：

```
=== kettle (...\models\obj_05.ply) ===
  candidates final: 20
   #1 kind=hole  Q=0.776 w=  9.5mm  feasible=True  pose#2   ← 侧把手
   #2 kind=hole  Q=0.732 w= 13.7mm  feasible=True  pose#3   ← 顶部拎手
   #3 kind=outer Q=0.663 w= 86.5mm  feasible=True  pose#4   ← 壶身
   ...
```

**排序应满足你在需求里描述的**：`侧把手 > 顶部拎手 > 壶身`；`ape` 应该没有 hole 类型的候选（本来就没有孔洞），只有一系列 outer 类型的躯干抓取，最高 Q 大约 0.9 左右。

如果排序完全对不上，请把 `outputs/preview_final_kettle.png` 和终端输出发我，我来定位问题。

---

## 5. 已知超参数取值

所有可调参数集中在 `hyperparams.yml` 中，每一条都附有中文/英文注释，说明：

- `[paper]` 标签：论文明确写死的数值（如 top-4 稳定姿态、上层 30% 阈值、d0 的 10% / 5%）
- `[decision]` 标签：论文未指定、由我自行选定或实测调过的（如 EFD 阶数=20、gripper 行程=110mm、摩擦系数=0.7、λ 权重分配等）——每一条都附有为什么这么选的注释

有 6 处偏离原文字面的实现决策已单独在 Phase 1 交付说明里写过，主要包括：稳定姿态引擎替换、高度阈值仅对封闭孔洞生效、hole 类型 d 的定义从"弧长"改为"pad 包裹余量"、gripper 行程放宽到 110mm、摩擦系数 μ=0.7、以及碰撞项延迟到 Phase 2 计算。这些的具体讨论见提交时的说明文字，源码里也有对应模块的 docstring 详解。

---

## 6. 常见问题

**Q: `ModuleNotFoundError: No module named 'pidenet_annotator'`**
A: 请确认你**在 `pidenet_annotator/` 这个目录下**运行 `python run_phase1.py`，而不是在它的父目录。或者在项目根目录 `pip install -e .` 一次即可从任何地方调用。

**Q: Windows 下路径以反斜杠 `\` 结尾会报错**
A: 例如 `--linemod E:\paper\PIDENet\LINEMOD\`（末尾有个 `\`），Python 的 argparse 会把 `\"` 当成转义符。解决办法：(1) 去掉末尾的 `\`；(2) 或者用正斜杠 `E:/paper/PIDENet/LINEMOD`；(3) 或者加引号 `"E:\paper\PIDENet\LINEMOD"`。这三种都可以。

**Q: 生成 `.html` 文件很慢/文件很大（5-6MB）**
A: 这是 Plotly 交互式 3D 图正常表现，把整个 mesh 顶点都嵌入了 HTML。如果不想要，注释掉 `run_phase1.py` 里 `interactive_html(...)` 那一行即可。

**Q: 我想要更多/更少的候选**
A: 有三个开关：
1. **过滤阈值** `--min-Q 0.1`（默认）：低于这个分数的候选不写入 yml、不画在图里、不出现在终端。设为 `0.0` 会保留全部（包括 feasible=False 的 Q=0 候选），设为 `0.7` 之类只保留高质量候选。
2. **显示上限** `--top-n N`：控制可视化图上最多画几个（不影响 yml 里保存的完整列表）。
3. **生成层面** `hyperparams.yml` 里：`outer_branch.max_pairs_per_pose`（每个姿态最多产生几个 outer 候选）、`outer_branch.min_depth_frac`（凸包缺陷最小深度阈值，越小越多）。

**Q: 排序不对，怎么办**
A: 首先看 `preview_final_kettle.png` 检查候选位置对不对；然后查 `phase1_candidates_kettle.yml` 里失败候选的 `component_scores` 各分量、`friction_angles_deg`、`feasible`。绝大多数排序偏差都是 gripper 行程范围（`w_max_mm`）或 μ 与你实际用的夹爪不匹配导致的。

---

## 7. 联系

changyuan087@gmail.com ，有任何问题请联系原作者

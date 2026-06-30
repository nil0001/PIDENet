# PIDENet  — LINEMOD -Grasp

目录

```
pidenet_repro/
├── README.md                          本文件
├── configs/
│   └── config.yaml                    所有路径与超参数（夹爪开口[0,100mm]等）
├── annotation/
│   ├── stable_pose.py                  3.2.1 稳定姿态分析（凸包+立体角）
│   ├── projection.py                   3.2.2 射线投影生成二值mask
│   ├── efd_contour.py                  3.2.3 连通域+EFD拟合+凸包缺陷抓取点提取
│   ├── approach_vector.py              3.2.4 反投影3D + PCA法向 + 坐标系构建
│   ├── grasp_scoring.py                3.2.5 四项几何打分 Q(g_k)
│   ├── collision_pruning.py            3.2.6 碰撞代理打分（D6说明的几何替代版）
│   └── pipeline.py                     串联以上所有步骤，物体坐标系候选抓取生成
├── utils/
│   ├── io_utils.py                     PLY/YAML/PNG读取，坐标系/单位自检
│   ├── geometry_utils.py               旋转矩阵、立体角、PCA等通用几何函数
│   └── transform.py                    物体系→相机系的位姿变换 (Eq. 22 思路)
├── visualize_3d.py                     物体坐标系下抓取候选的3D可视化（带CAD模型）
├── generate_camera_labels.py           主脚本：批量生成所有帧的相机系抓取标签 yml
└── run_object01_demo.py                一键运行脚本（先跑物体系标注+可视化，再生成所有帧标签）
```

## 运行方式

1. 修改 `configs/config.yaml` 中的 `dataset_root` 为你的本机路径，例如：
   ```yaml
   dataset_root: "D:/my_files/Linemod_preprocessed/Linemod_preprocessed"
   ```
2. 安装依赖：
   ```bash
   pip install numpy scipy opencv-python trimesh open3d pyefd pyyaml matplotlib
   ```
3. 运行：
   ```bash
   python run_object01_demo.py
   ```
4. 输出位于 `output/`：
   - `object_01_grasp_candidates_objframe.yml` — 物体坐标系下的抓取候选（含所有中间分数）
   - `object_01_grasp_candidates_3d.png` — 3D可视化静态图（同时会弹出 open3d 交互窗口）
   - `grasp_labels_01.yml` — 严格按你要求格式生成的相机坐标系全帧抓取标签

## 标签格式输出示例

```yaml
0:
  pose1: {w: 0.0823, v: [0.123, -0.045, 0.987], u: [0.991, 0.034, -0.125], center: [-0.012, 0.034, 0.512], S: 0.7421}
  pose2: {w: 0.0651, v: [...], u: [...], center: [...], S: 0.6103}
1:
  pose1: {...}
  ...
```

字段含义：
- `w`：夹爪宽度（米），对应论文 Eq.(2.1)节 $w_k = \|p_{k2}-p_{k1}\|$
- `v`：approach vector（单位向量），相机坐标系下
- `u`：orientation/rotation vector（单位向量），相机坐标系下，对应论文图2(b)中的旋转轴方向
- `center`：抓取中心点坐标（米），相机坐标系下
- `S`：最终抓取分数，$S = \tilde{Q}(g_k) = Q(g_k) \cdot P_{coll}(g_k)$（Eq. 11），**未做高斯扩散**，是离散候选点自身的分数

## 单位约定

LINEMOD 官方 PLY 模型单位通常为**毫米**，`cam_t_m2c` 同样为毫米，`depth.png` 中存储深度值通常以**毫米**为单位（需要乘 `depth_scale`，info.yml 中常为 1.0）。
本代码内部统一换算为**米**进行几何运算（相机系/物体系/可视化），输出标签的 `center` 和 `w` 字段均为**米**，请在使用时与你的机械臂控制单位制核对。代码会在运行时自动检测并打印 PLY 模型的尺度，便于核对。

## 已知环境兼容性问题

**open3d离屏截图在某些Linux容器/远程服务器环境下可能产生全黑图片**：这是部分机器在缺少独立显卡、依赖软件渲染(OSMesa)路径时的已知兼容性问题，开发过程中在沙箱容器里复现过该现象（即使是渲染最简单的单个球体也会得到全黑输出，证实与场景复杂度无关，纯粹是该特定渲染管线的限制）。代码已加入自动检测：若截图均值接近全黑会打印警告。

如果你在Windows本机（通常有正常显卡驱动）运行 `run_object01_demo.py` 遇到同样问题，请优先信任并查看同时生成的 `object_01_grasp_candidates_3d.png`（来自matplotlib，不依赖GPU渲染，在本项目所有测试环境中均稳定可用）；也可以将 `visualize_candidates_open3d` 的 `show_window` 参数设为 `True` 弹出交互式窗口直接查看（此路径走的是标准GUI渲染而非离屏渲染，通常不受此问题影响

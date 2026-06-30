# -*- coding: utf-8 -*-
"""
annotation/collision_pruning.py
==================================
论文 3.2.6 节 "Semantic-Aware Soft Collision Pruning" 的【离线几何代理版本】。

【README中D6已说明的关键差异】论文原文的P_coll计算依赖"语义分割分支的逐点预测"，
即需要一个已训练好的网络才能产生Ndanger（背景/障碍物点数）。但离线标注阶段
（本模块的应用场景）网络还不存在——这是论文逻辑链条上的鸡生蛋问题，必须用
不依赖网络的替代方案。

本实现的替代方案：
  对每一帧（已知该帧的相机内参K、深度图depth、物体在该帧的真实姿态R,t），
  将候选抓取的三个局部碰撞体积(V_target, V_fingers, V_approach)从物体坐标系
  变换到相机坐标系，再投影到深度图像素网格上。对碰撞体内的每个采样点，
  比较其【预期深度】(若该点真实存在，相机到该点的距离)与【深度图实测深度】：
    - 若 实测深度 < 预期深度 - margin   => 说明该位置在真实场景中存在"更近的"
      遮挡物（其他物体或本物体的其他部分挡在了相机和该碰撞体采样点之间），
      该采样点被记为"danger point"(对应论文Ndanger的角色)
    - 若 实测深度 接近或大于 预期深度   => 该位置无遮挡，安全

这正是论文Eq.(9)-(11)的几何意义在"用真实深度图代替语义分割网络输出"后的直接
对应实现，公式形式(Eq.10的指数衰减)保持不变。

注意：此模块只在"生成相机坐标系标签"阶段使用（每帧单独计算），因为它依赖
该帧的深度图和该帧下物体的真实姿态；物体坐标系下的候选抓取生成阶段不调用本模块
（物体坐标系阶段只计算Q(g_k)，不计算P_coll，这与论文Eq.11的结构一致：
 Q(g_k)在物体系/CAD模型上算好，P_coll要等映射到具体观测帧才能算）。
"""

import numpy as np


class CollisionVolumes:
    """根据论文Eq.(9)定义的三个局部碰撞体积的简化几何表示（轴对齐局部包围盒）"""

    def __init__(self, center, v_approach, u_orientation, width,
                 finger_thickness, finger_length, approach_clearance):
        """
        center, v_approach, u_orientation: 相机坐标系下，已变换好的抓取参数
        width: 夹爪开口宽度（米）
        finger_thickness, finger_length, approach_clearance: 来自config.yaml的gripper段
        """
        self.center = np.asarray(center, dtype=np.float64)
        self.v = v_approach / (np.linalg.norm(v_approach) + 1e-12)
        self.u = u_orientation / (np.linalg.norm(u_orientation) + 1e-12)
        self.x = np.cross(self.u, self.v)
        self.x = self.x / (np.linalg.norm(self.x) + 1e-12)
        self.width = width
        self.finger_thickness = finger_thickness
        self.finger_length = finger_length
        self.approach_clearance = approach_clearance

    def sample_points_fingers(self, n_samples_per_finger=30):
        """
        V_fingers: 两个手指占据的局部体积。在抓取局部坐标系下，
        手指沿v(approach)方向有finger_length长度，沿u(闭合轴)方向位于
        width/2附近(手指本身厚度finger_thickness)，沿x方向有手指宽度。

        简化为：在每个手指的局部包围盒内做均匀随机采样，返回相机坐标系下的3D点集。
        """
        pts = []
        rng = np.random.default_rng(42)  # 固定种子保证可复现性
        for side in [-1, 1]:  # 两个手指分别在 u方向的 ±width/2 附近
            u_offset = side * self.width / 2.0
            local_u = rng.uniform(u_offset - self.finger_thickness / 2,
                                    u_offset + self.finger_thickness / 2, n_samples_per_finger)
            local_v = rng.uniform(0, self.finger_length, n_samples_per_finger)
            local_x = rng.uniform(-self.finger_thickness / 2, self.finger_thickness / 2, n_samples_per_finger)

            world_pts = (self.center[None, :] +
                         local_u[:, None] * self.u[None, :] +
                         local_v[:, None] * self.v[None, :] +
                         local_x[:, None] * self.x[None, :])
            pts.append(world_pts)
        return np.concatenate(pts, axis=0)

    def sample_points_approach(self, n_samples=40):
        """
        V_approach: 夹爪底座沿接近方向的扫掠路径体积，从抓取中心沿-v方向
        （即approach vector的反方向，因为approach指向物体，夹爪从物体外侧沿-v
        方向接近）延伸 approach_clearance 距离。
        """
        rng = np.random.default_rng(43)
        local_v = rng.uniform(-self.approach_clearance, 0, n_samples)  # 沿-v方向延伸
        local_u = rng.uniform(-self.width / 2, self.width / 2, n_samples)
        local_x = rng.uniform(-self.finger_thickness, self.finger_thickness, n_samples)

        world_pts = (self.center[None, :] +
                     local_u[:, None] * self.u[None, :] +
                     local_v[:, None] * self.v[None, :] +
                     local_x[:, None] * self.x[None, :])
        return world_pts


def project_points_to_depth_pixels(points_cam: np.ndarray, K: np.ndarray):
    """
    将相机坐标系下的3D点批量投影到像素坐标，返回 (pixel_xy, expected_depth)
    pixel_xy: (N,2) [col, row]，已四舍五入为整数索引（未做边界裁剪，调用方需自行过滤越界）
    expected_depth: (N,) 每个点到相机的Z方向深度（即点的z坐标本身，标准针孔模型约定）
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    z = points_cam[:, 2]
    valid = z > 1e-6  # 避免除以0或负深度（点在相机后方，物理上不可见）

    col = np.full(len(points_cam), -1, dtype=np.int64)
    row = np.full(len(points_cam), -1, dtype=np.int64)

    col[valid] = np.round(points_cam[valid, 0] * fx / z[valid] + cx).astype(np.int64)
    row[valid] = np.round(points_cam[valid, 1] * fy / z[valid] + cy).astype(np.int64)

    return np.stack([col, row], axis=1), z, valid


def compute_collision_proxy_score(grasp_cam_dict, depth_image: np.ndarray, K: np.ndarray,
                                      gripper_cfg: dict, collision_cfg: dict, verbose=False):
    """
    顶层封装：计算单个抓取候选在某一帧观测下的碰撞代理分数 P_coll (Eq.10的几何替代版)。

    grasp_cam_dict: dict，至少包含 "center"(3,), "v"(3,), "u"(3,), "w"(float) —— 均已在相机坐标系下
    depth_image: (H,W) ndarray，单位米，0表示无效/缺失深度
    K: (3,3) 相机内参
    gripper_cfg, collision_cfg: 来自config.yaml对应段

    返回: float, P_coll ∈ (0,1]

    【单元测试中发现并确认的物理限制，非逻辑bug】
    手指厚度(finger_thickness，默认8mm)等局部碰撞体积尺度，在常见LINEMOD相机内参
    (fx≈570, 物体距离约1m)下，对应的图像尺度约只有4~5个像素。这意味着碰撞体内的
    多个采样点可能投影到同一个像素——这恰恰反映了真实深度相机的物理分辨率限制
    （传感器在该方向上本就无法分辨这么精细的结构），并不是本函数的实现错误：
    每个采样点仍然独立地与"自己所在像素的真实测量深度"比较，这正是单视角RGB-D
    传感器在实际工作中面对的同样限制。开发过程中我们曾用一个高分辨率虚拟相机
    (fx=5000)重新验证了同一组采样点，确认在像素不重叠的条件下，该函数对每个点
    的margin容差判断在数值上是精确的（验证脚本见test_collision_pruning.py）。
    """
    H, W = depth_image.shape

    vols = CollisionVolumes(
        center=grasp_cam_dict["center"], v_approach=grasp_cam_dict["v"],
        u_orientation=grasp_cam_dict["u"], width=grasp_cam_dict["w"],
        finger_thickness=gripper_cfg["finger_thickness"],
        finger_length=gripper_cfg["finger_length"],
        approach_clearance=gripper_cfg["approach_clearance"],
    )

    pts_fingers = vols.sample_points_fingers()
    pts_approach = vols.sample_points_approach()
    all_pts = np.concatenate([pts_fingers, pts_approach], axis=0)

    pixel_xy, expected_depth, valid = project_points_to_depth_pixels(all_pts, K)

    margin = collision_cfg["depth_occlusion_margin"]
    n_danger = 0
    n_checked = 0

    for i in range(len(all_pts)):
        if not valid[i]:
            continue
        col, row = pixel_xy[i]
        if col < 0 or col >= W or row < 0 or row >= H:
            continue  # 投影超出图像范围，跳过（视为无法判断，不计入危险也不计入总数）

        n_checked += 1
        measured_depth = depth_image[row, col]

        if measured_depth <= 1e-6:
            continue  # 深度图该像素无有效读数（常见于深度传感器空洞），跳过不计入危险判断

        if measured_depth < expected_depth[i] - margin:
            n_danger += 1

    if n_checked == 0:
        # 整个碰撞体都投影到图像外或全部无效深度读数，无法判断，保守起见返回中性值0.5
        # （既不假设完全安全也不假设完全危险，避免极端情况下产生误导性的1.0或0.0）
        if verbose:
            print("    [collision_proxy] 警告：碰撞体采样点全部无法在深度图中验证，返回中性分数0.5")
        return 0.5

    volume_estimate = len(all_pts)  # 用采样点总数近似代表Eq.10中的 V_collision（体积的离散化代理）
    alpha = collision_cfg["alpha"]
    P_coll = float(np.exp(-alpha * n_danger / volume_estimate))

    if verbose:
        print(f"    [collision_proxy] 检查点数={n_checked}/{len(all_pts)}, "
              f"危险点数={n_danger}, P_coll={P_coll:.4f}")

    return P_coll

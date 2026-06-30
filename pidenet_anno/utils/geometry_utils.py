# -*- coding: utf-8 -*-
"""
utils/geometry_utils.py
========================
通用几何计算函数：立体角估计、PCA法向估计、Gram-Schmidt正交化、
点到直线距离等。被 stable_pose.py / approach_vector.py / grasp_scoring.py 复用。
"""

import numpy as np


# ==============================================================================
# 立体角估计 (论文 3.2.1, Eq.(1) 中的 W_k)
# ==============================================================================
def triangle_solid_angle(apex: np.ndarray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> float:
    """
    计算从顶点 apex 看三角形 (v0, v1, v2) 所张的立体角（单位：球面度，sr）。
    使用 Van Oosterom-Strackee 公式：

        tan(Ω/2) = |a·(b×c)| / (|a||b||c| + (a·b)|c| + (b·c)|a| + (c·a)|b|)

    其中 a = v0-apex, b = v1-apex, c = v2-apex
    这是计算球面三角形/三棱锥立体角的标准闭式解，数值稳定。
    """
    a = v0 - apex
    b = v1 - apex
    c = v2 - apex

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    c_norm = np.linalg.norm(c)

    if a_norm < 1e-12 or b_norm < 1e-12 or c_norm < 1e-12:
        return 0.0

    numerator = np.abs(np.dot(a, np.cross(b, c)))
    denominator = (
        a_norm * b_norm * c_norm
        + np.dot(a, b) * c_norm
        + np.dot(b, c) * a_norm
        + np.dot(c, a) * b_norm
    )

    if denominator <= 0:
        # 钝角情况，立体角 > 2π，需要特殊处理
        omega = 2 * np.pi - 2 * np.arctan2(numerator, -denominator) if numerator > 0 else 2 * np.pi
    else:
        omega = 2 * np.arctan2(numerator, denominator)

    return float(omega)


def polygon_solid_angle_from_apex(apex: np.ndarray, polygon_vertices: np.ndarray) -> float:
    """
    计算顶点 apex 相对一个（可能非三角形的）平面多边形支撑面所张的总立体角。

    做法：以多边形质心为扇形中心，将多边形三角化为若干个三角形扇区，
    对每个三角形调用 triangle_solid_angle 后求和。

    对应论文描述："以投影点为顶点、支撑面边界为底，计算质心相对支撑面的立体角"——
    论文原文这里的几何关系略有歧义（"投影点为顶点"和"质心的立体角"两个描述在严格意义上
    指代了不同几何体），本函数采用更直接且物理意义明确的版本：
        以物体质心(3D)为顶点(apex)，对支撑面多边形(凸包的一个面)进行三角化后
        计算质心对该支撑面的总立体角。
    这正是Goldberg等人在抓取稳定性分析中"valence angle/solid angle"的标准定义：
    立体角越大，物体翻倒离开当前支撑面所需跨越的角度越大，姿态越稳定。
    """
    if len(polygon_vertices) < 3:
        return 0.0

    centroid = polygon_vertices.mean(axis=0)
    total = 0.0
    n = len(polygon_vertices)
    for i in range(n):
        v0 = polygon_vertices[i]
        v1 = polygon_vertices[(i + 1) % n]
        total += triangle_solid_angle(apex, centroid, v0, v1)
    return total


# ==============================================================================
# PCA 法向估计 (论文 3.2.4)
# ==============================================================================
def pca_normal_estimation(points: np.ndarray, reference_outward=None):
    """
    对一组3D邻域点做PCA，返回最小特征值对应的特征向量（局部法向估计）。

    points: (N, 3) ndarray
    reference_outward: 若提供一个参考"应朝外"方向（如 质心->采样点 的方向），
                        用于消解PCA法向±符号的歧义性（PCA本身给出的法向方向不确定）

    返回: (3,) 单位法向向量
    """
    if len(points) < 3:
        raise ValueError(f"PCA法向估计至少需要3个点，当前只有{len(points)}个")

    centered = points - points.mean(axis=0)
    cov = centered.T @ centered / len(points)
    eigvals, eigvecs = np.linalg.eigh(cov)  # 升序排列
    normal = eigvecs[:, 0]  # 最小特征值对应的特征向量 = 法向（曲率最小变化方向的正交方向）
    normal = normal / (np.linalg.norm(normal) + 1e-12)

    if reference_outward is not None:
        ref = reference_outward / (np.linalg.norm(reference_outward) + 1e-12)
        if np.dot(normal, ref) < 0:
            normal = -normal

    return normal


# ==============================================================================
# Gram-Schmidt 正交化 (论文 3.3 节提到，用于聚合后的旋转矩阵修正)
# ==============================================================================
def gram_schmidt_orthogonalize(v_approach: np.ndarray, u_orientation: np.ndarray):
    """
    给定approach向量v和orientation向量u（可能不完全垂直，例如聚合/平均后），
    执行Gram-Schmidt正交化，返回严格正交的 (v_ortho, u_ortho, x_ortho) 构成右手系。

    构造逻辑对应论文 Eq.(3): x = y × z 的右手系规则，
    这里令 v=z(approach,接近方向), u=y(orientation,闭合轴方向)，
    与论文图2(b)中心坐标系标注一致（v为接近向量，对应z；u为旋转/闭合向量，对应论文公式里的y）。
    """
    v = v_approach / (np.linalg.norm(v_approach) + 1e-12)
    u_proj = u_orientation - np.dot(u_orientation, v) * v
    u = u_proj / (np.linalg.norm(u_proj) + 1e-12)
    x = np.cross(u, v)
    x = x / (np.linalg.norm(x) + 1e-12)
    return v, u, x


def rotation_matrix_from_vectors(v_approach: np.ndarray, u_orientation: np.ndarray) -> np.ndarray:
    """
    由正交化后的 approach(v) 和 orientation(u) 向量构造3x3旋转矩阵 R = [x, u, v]
    （列向量分别为抓取坐标系的x/y/z轴在世界/相机坐标系下的表示）。
    """
    v, u, x = gram_schmidt_orthogonalize(v_approach, u_orientation)
    R = np.stack([x, u, v], axis=1)  # 按列拼接
    return R


# ==============================================================================
# 点到直线的垂直距离 (论文 Eq.(7) 中 d_hat 的计算)
# ==============================================================================
def point_to_line_distance(point: np.ndarray, line_point: np.ndarray, line_dir: np.ndarray) -> float:
    """
    计算 point 到由 (line_point, line_dir) 定义的直线的垂直距离。
    line_dir 不需要预先归一化，函数内部会归一化。
    """
    d = line_dir / (np.linalg.norm(line_dir) + 1e-12)
    w = point - line_point
    proj_len = np.dot(w, d)
    perp_vec = w - proj_len * d
    return float(np.linalg.norm(perp_vec))


# ==============================================================================
# 友好型摩擦锥角度检查 (论文 Eq.(8) 中的可行性指示函数 I 用到)
# ==============================================================================
def within_friction_cone(grasp_axis: np.ndarray, normal: np.ndarray, mu: float) -> bool:
    """
    检查抓取闭合轴与某接触点法向之间的夹角是否严格小于摩擦锥半角 arctan(mu)。
    grasp_axis, normal 均会被内部归一化。

    注意：这里【不】对点积取绝对值。物理上，闭合轴方向必须与"指向物体内部"的法向
    （即 -outward_normal，因为手指要把物体往里压才能产生有效正压力）基本同向，
    若 grasp_axis 与 outward_normal 夹角接近180°（即与-normal同向），才说明手指
    朝物体内部施压，是合法的对极抓取方向；若 grasp_axis 与 outward_normal 同向，
    说明手指在"往外拉"，不可能产生正压力，必须判定为不合法。

    调用约定：本函数期望传入的 normal 是【接触点外法向 (outward normal)】，
    函数内部自动取其反向（即物体内部方向）与 grasp_axis 比较夹角。
    """
    a = grasp_axis / (np.linalg.norm(grasp_axis) + 1e-12)
    n_outward = normal / (np.linalg.norm(normal) + 1e-12)
    n_inward = -n_outward
    cos_angle = np.clip(np.dot(a, n_inward), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    cone_half_angle = np.arctan(mu)
    return angle < cone_half_angle

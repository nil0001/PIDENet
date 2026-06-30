# -*- coding: utf-8 -*-
"""
annotation/grasp_scoring.py
=============================
论文 3.2.5 节 "Grasp Quality Scoring" 的实现，严格对应公式 (4)-(8)。

四项打分：
  S_geo  Eq.(4): tanh(d/d0)                       几何嵌合度（凹陷越深越稳）
  S_ali  Eq.(5): 0.5*(|u·n1| + |u·n2|)             对极对齐度
  S_wid  Eq.(6): exp(-(w-w*)^2 / (2*sigma_w^2))     宽度合规度
  S_com  Eq.(7): exp(-d_hat^2 / (2*sigma_c^2))      力矩臂打分（离CoM越近越稳）

最终: Q(g_k) = I( sum(lambda_i * S_i) )            Eq.(8)
其中 I 是可行性门控：摩擦锥约束 + 物理合法性检查。
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.geometry_utils import point_to_line_distance, within_friction_cone


class GraspScore:
    """单个抓取候选的完整打分明细"""
    def __init__(self, S_geo, S_ali, S_wid, S_com, feasible, Q):
        self.S_geo = S_geo
        self.S_ali = S_ali
        self.S_wid = S_wid
        self.S_com = S_com
        self.feasible = feasible   # 可行性门控 I 的结果 (bool)
        self.Q = Q                  # 最终分数 Q(g_k)，若feasible=False则Q=0

    def __repr__(self):
        return (f"GraspScore(Q={self.Q:.4f}, feasible={self.feasible}, "
                f"S_geo={self.S_geo:.3f}, S_ali={self.S_ali:.3f}, "
                f"S_wid={self.S_wid:.3f}, S_com={self.S_com:.3f})")


def compute_geometric_interlocking_score(depth_value: float, d0: float, is_hole_grasp=False) -> float:
    """
    Eq.(4): S_geo = tanh(d/d0) ∈ [0,1)

    depth_value: 论文原文区分两种情形：
      - 外轮廓抓取(outer_defect): d = 凸性缺陷深度（米）
      - 内孔抓取(inner_hole): d 替换为"夹爪环抱的孔壁弧段所张的中心角"（弧度）
        本实现对inner_hole情形，调用方应预先将弧度值传入此函数的depth_value参数，
        d0也应使用一个角度量级的归一化常数（而非长度），调用方负责保证量纲一致。
    """
    if d0 <= 0:
        raise ValueError("d0必须为正数")
    return float(np.tanh(depth_value / d0))


def compute_antipodal_alignment_score(u_closing_axis: np.ndarray,
                                        normal1: np.ndarray, normal2: np.ndarray) -> float:
    """
    Eq.(5): S_ali = 0.5 * (|u·n1| + |u·n2|)

    u_closing_axis: 归一化的闭合轴方向 (p_k2-p_k1)/||p_k2-p_k1||
    normal1, normal2: 两个接触点的outward法向（单位向量）

    注：论文公式本身对法向取了绝对值|u·n|，这意味着该项打分对"闭合轴与法向平行"
    本身是对称的（不区分指向内还是指向外），物理意义是"闭合轴与表面法线越平行，
    越接近理想对极抓取(antipodal grasp)，越不容易在闭合过程中打滑"。
    这与下方compute_feasibility_gate里的摩擦锥方向性判断（必须指向内部产生压力）
    是两个独立的判断维度：S_ali只评估"对齐程度"这个软指标，方向性的硬约束交给可行性门控。
    """
    u = u_closing_axis / (np.linalg.norm(u_closing_axis) + 1e-12)
    n1 = normal1 / (np.linalg.norm(normal1) + 1e-12)
    n2 = normal2 / (np.linalg.norm(normal2) + 1e-12)
    return float(0.5 * (abs(np.dot(u, n1)) + abs(np.dot(u, n2))))


def compute_width_compatibility_score(width: float, w_max: float, gamma: float, sigma_w: float) -> float:
    """
    Eq.(6): S_wid = exp( -(w - w*)^2 / (2*sigma_w^2) ),  w* = gamma * w_max
    """
    w_star = gamma * w_max
    return float(np.exp(-((width - w_star) ** 2) / (2 * sigma_w ** 2)))


def compute_moment_arm_score(com: np.ndarray, grasp_center: np.ndarray,
                                closing_axis: np.ndarray, sigma_c: float) -> float:
    """
    Eq.(7): S_com = exp( -d_hat^2 / (2*sigma_c^2) )
    d_hat: 物体质心(CoM)到"闭合轴所在直线"的垂直距离
    """
    d_hat = point_to_line_distance(com, grasp_center, closing_axis)
    return float(np.exp(-(d_hat ** 2) / (2 * sigma_c ** 2)))


def compute_feasibility_gate(u_closing_axis: np.ndarray, normal1: np.ndarray, normal2: np.ndarray,
                                width: float, w_min: float, w_max: float, mu: float) -> bool:
    """
    Eq.(8)中的可行性指示函数 I：
      - 闭合轴与两个接触点法向的夹角都必须严格小于摩擦锥半角 arctan(mu)（动力学合法性）
      - 夹爪宽度必须落在物理可达范围 [w_min, w_max] 内

    【关键物理约束 — 闭合轴方向性】
    u_closing_axis 的定义是从p_k1指向p_k2的方向（见approach_vector.py: y_axis =
    (p2-center)/||...||，等价于(p2-p1)方向）。但并联夹爪的两个手指是【相向运动】夹紧物体的：
      - 接触点p1处的手指运动方向 = +u_closing_axis（朝p2方向，即朝物体内部）
      - 接触点p2处的手指运动方向 = -u_closing_axis（朝p1方向，即朝物体内部）
    因此判断"手指能否在该接触点产生有效正压力"时，p1和p2必须分别用相反符号的轴方向去检查，
    不能用同一个u对两点做同方向判断——这是本实现early版本中的一个真实bug，已通过手算
    反例(见test_grasp_scoring.py)发现并在此修正。

    within_friction_cone内部约定：传入的法向是outward法向，函数内部自动取反(指向物体内部)
    与传入的axis方向比较夹角。所以这里对p1传入(+u)，对p2传入(-u)。
    """
    if not (w_min <= width <= w_max):
        return False

    u = u_closing_axis / (np.linalg.norm(u_closing_axis) + 1e-12)

    cone_ok_1 = within_friction_cone(u, normal1, mu)        # p1手指运动方向 = +u
    cone_ok_2 = within_friction_cone(-u, normal2, mu)        # p2手指运动方向 = -u

    return bool(cone_ok_1 and cone_ok_2)


def score_grasp_candidate(geom, com: np.ndarray, depth_value: float, d0: float,
                            scoring_cfg: dict, gripper_cfg: dict, is_hole_grasp=False) -> GraspScore:
    """
    顶层封装：给定一个GraspGeometry3D对象(来自approach_vector.py)和物体质心，
    计算完整的四项打分+可行性门控，返回最终Q(g_k)。

    geom: GraspGeometry3D 对象，需含 .u(闭合轴/orientation), .normal1, .normal2,
          .width, .center
    com: (3,) 物体质心坐标（与geom同坐标系）
    depth_value: 几何嵌合度计算所需的深度值（凸性缺陷深度 或 孔壁弧度，米/弧度）
    d0: Eq.(4)归一化常数
    scoring_cfg: config.yaml中的scoring段（已解析null值为实际数值后传入）
    gripper_cfg: config.yaml中的gripper段
    """
    lambda_weights = scoring_cfg["lambda_weights"]
    assert abs(sum(lambda_weights) - 1.0) < 1e-6, \
        f"lambda权重之和必须为1，当前为{sum(lambda_weights)}"

    S_geo = compute_geometric_interlocking_score(depth_value, d0, is_hole_grasp)
    S_ali = compute_antipodal_alignment_score(geom.u, geom.normal1, geom.normal2)
    S_wid = compute_width_compatibility_score(
        geom.width, gripper_cfg["w_max"], scoring_cfg["gamma_width"], scoring_cfg["sigma_w"]
    )
    S_com = compute_moment_arm_score(com, geom.center, geom.u, scoring_cfg["sigma_c"])

    feasible = compute_feasibility_gate(
        geom.u, geom.normal1, geom.normal2,
        geom.width, gripper_cfg["w_min"], gripper_cfg["w_max"], gripper_cfg["friction_coeff"]
    )

    if feasible:
        Q = (lambda_weights[0] * S_geo + lambda_weights[1] * S_ali +
             lambda_weights[2] * S_wid + lambda_weights[3] * S_com)
    else:
        Q = 0.0

    return GraspScore(S_geo=S_geo, S_ali=S_ali, S_wid=S_wid, S_com=S_com,
                        feasible=feasible, Q=Q)

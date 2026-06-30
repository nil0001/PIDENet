# -*- coding: utf-8 -*-
"""
utils/transform.py
====================
物体坐标系 <-> 相机坐标系 的刚体变换工具。

对应论文 Section 4.3, Eq.(22):
    p_cam = R @ p_obj + t
    v_cam = R @ v_obj          (方向向量只受旋转影响，不叠加平移)

其中 R, t 来自 LINEMOD gt.yml 中的 cam_R_m2c (3x3), cam_t_m2c (3,)，
含义是"物体坐标系下的点经过该变换后落在相机坐标系下的坐标"，
即 R, t 描述的是 model-to-camera (m2c) 变换，与论文Eq.22的R,t语义一致。
"""

import numpy as np


def transform_point_obj_to_cam(p_obj: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    p_obj: (3,) 物体坐标系下的点坐标（米）
    R: (3,3) cam_R_m2c
    t: (3,) cam_t_m2c（米）
    返回: (3,) 相机坐标系下的点坐标（米）
    """
    return R @ p_obj + t


def transform_vector_obj_to_cam(v_obj: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    方向向量（approach vector / orientation vector）只经过旋转，不叠加平移。
    返回的向量会重新归一化为单位向量（防止数值误差累积导致模长漂移）。
    """
    v_cam = R @ v_obj
    norm = np.linalg.norm(v_cam)
    if norm < 1e-12:
        raise ValueError("变换后的方向向量模长接近0，输入向量可能有问题")
    return v_cam / norm


def transform_grasp_obj_to_cam(grasp_obj: dict, R: np.ndarray, t: np.ndarray) -> dict:
    """
    将物体坐标系下的一个完整抓取候选 (center, v, u, w, 各类分数) 变换到相机坐标系。

    grasp_obj 期望包含键: "center"(3,), "v"(3,), "u"(3,), "w"(float),
                          以及任意分数字段（直接原样保留，分数是标量、坐标系无关）

    返回一个新dict，坐标/向量字段已转换为相机坐标系，其余字段（标量分数等）原样拷贝。
    """
    out = dict(grasp_obj)  # 浅拷贝，标量字段直接保留
    out["center"] = transform_point_obj_to_cam(grasp_obj["center"], R, t)
    out["v"] = transform_vector_obj_to_cam(grasp_obj["v"], R)
    out["u"] = transform_vector_obj_to_cam(grasp_obj["u"], R)
    # w（夹爪宽度）是标量距离，刚体旋转+平移不改变两点间距离，原样保留
    out["w"] = grasp_obj["w"]
    return out


def batch_transform_grasps_obj_to_cam(grasp_list_obj: list, R: np.ndarray, t: np.ndarray) -> list:
    """对一组候选抓取批量做坐标变换，返回新列表"""
    return [transform_grasp_obj_to_cam(g, R, t) for g in grasp_list_obj]

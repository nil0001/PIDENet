# -*- coding: utf-8 -*-
"""
annotation/pipeline.py
========================
顶层流水线：串联论文3.2.1~3.2.5节，从CAD模型生成物体坐标系下的候选抓取列表。

流程：
  CAD模型 (mesh)
    -> stable_pose.analyze_stable_poses()          [3.2.1] Top-4稳定姿态
    -> 对每个姿态:
         projection.raycast_orthographic_projection() [3.2.2] 二值mask
         efd_contour.extract_all_grasp_pairs()         [3.2.3] 2D抓取点对
         approach_vector.backproject_grasp_pairs_to_3d() + build_grasp_geometry()
                                                         [3.2.4] 3D几何(在投影坐标系下)
         approach_vector.rotate_geometry_back_to_object_frame() 变换回物体坐标系
         grasp_scoring.score_grasp_candidate()          [3.2.5] 打分Q(g_k)
    -> 汇总所有姿态产出的候选，按Q降序排列
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from annotation.stable_pose import analyze_stable_poses
from annotation.projection import raycast_orthographic_projection
from annotation.efd_contour import extract_all_grasp_pairs
from annotation.approach_vector import (
    backproject_grasp_pairs_to_3d, build_grasp_geometry, rotate_geometry_back_to_object_frame
)
from annotation.grasp_scoring import score_grasp_candidate
from scipy.spatial import cKDTree


class GraspCandidate:
    """一个完整的物体坐标系候选抓取，整合几何信息和打分明细"""
    def __init__(self, geom, score, source_pose_id, source_type):
        self.geom = geom              # GraspGeometry3D, 物体坐标系下
        self.score = score              # GraspScore
        self.source_pose_id = source_pose_id  # 来自哪个稳定姿态(0~3)
        self.source_type = source_type        # "outer_defect" 或 "inner_hole"

    def __repr__(self):
        return (f"GraspCandidate(Q={self.score.Q:.4f}, pose_id={self.source_pose_id}, "
                f"type={self.source_type}, width={self.geom.width*1000:.1f}mm)")


def _resolve_scoring_constants(scoring_cfg: dict, mesh, top_k_poses_results):
    """
    将config.yaml中标记为null的自适应常数(d0, sigma_w, sigma_c)解析为具体数值。
    论文未给出这些常数的固定值，本实现按config.yaml中注释说明的规则自适应计算
    （详见README.md的说明表格）。
    """
    resolved = dict(scoring_cfg)
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))  # 物体特征尺寸D

    if resolved["d0_geo"] is None:
        resolved["d0_geo"] = 0.1 * diag
    if resolved["sigma_w"] is None:
        # 注意：sigma_w依赖w_max-w_min，这里在调用方(generate_object_candidates)注入gripper_cfg后計算
        pass  # 占位，实际在下方函数体内结合gripper_cfg解析
    if resolved["sigma_c"] is None:
        resolved["sigma_c"] = 0.25 * diag

    resolved["_object_diag"] = diag
    return resolved


def generate_object_frame_candidates(mesh, config: dict, verbose=True):
    """
    顶层函数：给定CAD模型(米单位)和完整config字典，生成物体坐标系下所有候选抓取。

    返回: list[GraspCandidate]，按Q降序排列
    """
    stable_pose_cfg = config["stable_pose"]
    projection_cfg = config["projection"]
    efd_cfg = config["efd"]
    scoring_cfg_raw = config["scoring"]
    gripper_cfg = config["gripper"]

    # 物体质心（用于Eq.7力矩臂打分），watertight时用真实体积质心，否则回退顶点平均
    if mesh.is_watertight:
        com = np.asarray(mesh.center_mass)
    else:
        com = mesh.vertices.mean(axis=0)
        if verbose:
            print("[pipeline] ⚠ 网格非水密，质心回退为顶点几何平均值")

    if verbose:
        print("=" * 70)
        print("[Step 1/5] 稳定姿态分析 (论文3.2.1节)")
    stable_poses = analyze_stable_poses(mesh, top_k=stable_pose_cfg["top_k"], verbose=verbose)

    scoring_cfg = _resolve_scoring_constants(scoring_cfg_raw, mesh, stable_poses)
    sigma_w_resolved = scoring_cfg["sigma_w"]
    if sigma_w_resolved is None:
        sigma_w_resolved = 0.15 * (gripper_cfg["w_max"] - gripper_cfg["w_min"])
    scoring_cfg["sigma_w"] = sigma_w_resolved

    if verbose:
        print(f"\n[pipeline] 自适应打分常数解析结果: d0={scoring_cfg['d0_geo']*1000:.2f}mm, "
              f"sigma_w={scoring_cfg['sigma_w']*1000:.2f}mm, sigma_c={scoring_cfg['sigma_c']*1000:.2f}mm "
              f"(物体特征尺寸D={scoring_cfg['_object_diag']*1000:.1f}mm)")

    all_candidates = []

    for pose_id, stable_pose in enumerate(stable_poses):
        if verbose:
            print(f"\n{'='*70}")
            print(f"[Step 2-5] 处理稳定姿态 #{pose_id} (face_id={stable_pose.face_id}, "
                  f"P_k={stable_pose.probability*100:.1f}%)")
            print(f"  [Step 2/5] 射线投影生成mask (论文3.2.2节)")

        try:
            proj_result = raycast_orthographic_projection(
                mesh, stable_pose.rotation_to_zup,
                resolution=projection_cfg["resolution"],
                thin_wall_height_ratio=projection_cfg["thin_wall_height_ratio"],
                cavity_pixel_ratio_threshold=projection_cfg["cavity_pixel_ratio_threshold"],
                verbose=verbose,
            )
        except Exception as e:
            if verbose:
                print(f"  ⚠ 该姿态投影失败: {e}，跳过此姿态")
            continue

        if verbose:
            print(f"  [Step 3/5] EFD轮廓拟合+抓取点提取 (论文3.2.3节)")
        try:
            pairs_2d, smooth_outer = extract_all_grasp_pairs(proj_result.mask, efd_cfg, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"  ⚠ 该姿态轮廓分析失败: {e}，跳过此姿态")
            continue

        if len(pairs_2d) == 0:
            if verbose:
                print(f"  该姿态未提取到任何有效抓取点对，跳过")
            continue

        if verbose:
            print(f"  [Step 4/5] 反投影3D + 构建approach vector (论文3.2.4节)")
        pairs_3d_raw = backproject_grasp_pairs_to_3d(
            pairs_2d, proj_result, proj_result.pixel_size, proj_result.origin_xy, verbose=verbose
        )

        if len(pairs_3d_raw) == 0:
            if verbose:
                print(f"  该姿态所有2D点反投影3D均失败，跳过")
            continue

        rotated_mesh_vertices = (stable_pose.rotation_to_zup @ mesh.vertices.T).T
        kdtree_rotated_verts = cKDTree(rotated_mesh_vertices)

        # build_grasp_geometry内部调用estimate_local_normal_pca，该函数在检测到PCA退化
        # (邻域点近似共面/共线，常见于细长结构在低密度网格上的KNN查询)时，会回退到基于
        # mesh.face_normals + mesh.triangles_center的备选方案。因此这里不能只构造一个仅有
        # .vertices属性的轻量替身——必须提供完整的旋转后面法向和面中心信息，否则退化检测
        # 触发时会因缺少属性而出错（这是开发过程中发现并修复的真实问题：早期版本只传了
        # .vertices，在测试细颈/把手等高曲率结构时，PCA退化的样本因无法回退而退化估计未被
        # 纠正，导致法向严重失真，进而让可行性门控几乎全部判定为不可行）。
        # 面法向是方向量，旋转矩阵直接作用；面中心是点坐标，同样直接施加旋转(无平移，因为
        # stable_pose.py的rotation_to_zup本身是绕原点的纯旋转，不含平移分量)。
        rotated_face_normals = (stable_pose.rotation_to_zup @ mesh.face_normals.T).T
        rotated_triangles_center = (stable_pose.rotation_to_zup @ mesh.triangles_center.T).T

        class _RotatedMeshView:
            def __init__(self, vertices, face_normals, triangles_center):
                self.vertices = vertices
                self.face_normals = face_normals
                self.triangles_center = triangles_center

        rotated_mesh_view = _RotatedMeshView(
            rotated_mesh_vertices, rotated_face_normals, rotated_triangles_center
        )

        if verbose:
            print(f"  [Step 5/5] 打分 Q(g_k) (论文3.2.5节)")

        pose_candidates = []
        for pair_3d in pairs_3d_raw:
            try:
                geom_rotated = build_grasp_geometry(
                    pair_3d["p1_3d"], pair_3d["p2_3d"], rotated_mesh_view,
                    k_neighbors=15, kdtree_mesh_vertices=kdtree_rotated_verts
                )
            except ValueError:
                continue  # 接触点几乎重合，跳过

            geom_obj_frame = rotate_geometry_back_to_object_frame(geom_rotated, stable_pose.rotation_to_zup)

            if pair_3d["source"] == "outer_defect":
                # 几何嵌合度的depth_value: 凸性缺陷深度(像素) -> 转换为物理长度(米)
                depth_value = pair_3d["defect_depth_px"] * proj_result.pixel_size
                is_hole = False
            else:
                # 内孔抓取: 论文用"夹爪环抱孔壁弧段所张的中心角"代替深度，
                # 简化实现：用 width/(孔洞半径估计) 这一比值的arcsin近似作为角度代理
                # （注：论文未给出该角度的精确几何定义，这是本实现的工程近似，README已说明）
                depth_value = geom_obj_frame.width  # 简化：直接用宽度作为该项的代理量纲
                is_hole = True

            score = score_grasp_candidate(
                geom_obj_frame, com, depth_value, scoring_cfg["d0_geo"],
                scoring_cfg, gripper_cfg, is_hole_grasp=is_hole
            )

            pose_candidates.append(GraspCandidate(
                geom=geom_obj_frame, score=score,
                source_pose_id=pose_id, source_type=pair_3d["source"]
            ))

        if verbose:
            feasible_count = sum(1 for c in pose_candidates if c.score.feasible)
            print(f"  该姿态产出候选数={len(pose_candidates)}, 其中可行(feasible)={feasible_count}")

        all_candidates.extend(pose_candidates)

    all_candidates.sort(key=lambda c: c.score.Q, reverse=True)

    if verbose:
        print(f"\n{'='*70}")
        print(f"[pipeline完成] 总候选数={len(all_candidates)}, "
              f"可行候选数={sum(1 for c in all_candidates if c.score.feasible)}")
        print(f"Top-5候选预览:")
        for c in all_candidates[:5]:
            print(f"  {c}")

    return all_candidates

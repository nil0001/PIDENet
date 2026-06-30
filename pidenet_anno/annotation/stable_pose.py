# -*- coding: utf-8 -*-
"""
annotation/stable_pose.py
==========================
论文 3.2.1 节 "Physical Stable Pose Analysis" 的实现。

算法流程：
  1. 计算CAD模型的全局凸包
  2. 对凸包的每一个面（作为候选支撑面），将物体质心投影到该面所在平面，
     计算质心相对该支撑面边界的立体角 W_k
  3. 按 Eq.(1) 归一化得到每个支撑面的"静止概率" P_k
  4. 取概率最高的Top-4个支撑面，作为后续投影分析的候选姿态基准 K*

【实现细节说明】
凸包的每个三角面本身面积通常很小（trimesh的凸包是三角化网格），若直接以单个
三角面为"支撑面"，会产生大量几乎共面的相邻小三角面，物理上它们应该被当作同一个
"支撑面"看待（例如一个矩形底面会被三角化成2个三角形）。因此本实现先对凸包面
按法向相似性做合并聚类（法向夹角 < 阈值 且 相邻 视为同一支撑面），再对合并后的
"宏观支撑面多边形"计算立体角，这样得到的Top-4才是物理上有意义的、彼此明显不同
的姿态，而不是被三角化噪声分割出来的虚假候选。
"""

import numpy as np
from scipy.spatial import ConvexHull
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.geometry_utils import polygon_solid_angle_from_apex


class StablePoseResult:
    """单个候选稳定姿态的结果容器"""
    def __init__(self, face_id, normal, plane_point, polygon_vertices_3d,
                 solid_angle, probability, rotation_to_zup):
        self.face_id = face_id
        self.normal = normal                      # 支撑面外法向（指向物体外部，即重力方向的反方向）
        self.plane_point = plane_point             # 支撑面上一点（用于定义平面）
        self.polygon_vertices_3d = polygon_vertices_3d  # 支撑面边界多边形顶点（原始物体坐标系下）
        self.solid_angle = solid_angle             # W_k
        self.probability = probability             # P_k
        self.rotation_to_zup = rotation_to_zup     # 将该支撑面法向旋转对齐到+Z轴的旋转矩阵
                                                    # （用于3.2.2节的射线投影：投影沿世界Z轴向下，
                                                    #  需要先把物体"摆正"到该支撑面朝下的姿态）

    def __repr__(self):
        return (f"StablePoseResult(face_id={self.face_id}, W_k={self.solid_angle:.4f}, "
                f"P_k={self.probability:.4f})")


def _merge_coplanar_faces(hull: ConvexHull, vertices: np.ndarray, angle_thresh_deg=3.0):
    """
    将凸包三角面按法向相似性+空间邻接合并为"宏观支撑面"。

    返回: list of dict，每个dict包含:
        - "normal": 合并后面的平均法向
        - "vertices_3d": 该宏观面边界的有序多边形顶点(沿边界一圈，已去重)
        - "plane_point": 平面上一点
        - "triangle_ids": 原始三角面id列表（用于面积加权等）
    """
    face_normals = hull.equations[:, :3]  # ConvexHull.equations: [a,b,c,d] for ax+by+cz+d=0, 法向已归一化
    n_faces = len(hull.simplices)

    # 并查集做连通合并
    parent = list(range(n_faces))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # 构建三角面之间的邻接关系（共享一条边即为邻接）
    edge_to_faces = {}
    for fi, simplex in enumerate(hull.simplices):
        for k in range(3):
            v_a, v_b = simplex[k], simplex[(k + 1) % 3]
            edge = (min(v_a, v_b), max(v_a, v_b))
            edge_to_faces.setdefault(edge, []).append(fi)

    angle_thresh_rad = np.deg2rad(angle_thresh_deg)
    for edge, faces in edge_to_faces.items():
        if len(faces) == 2:
            f1, f2 = faces
            n1, n2 = face_normals[f1], face_normals[f2]
            cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            if angle < angle_thresh_rad:
                union(f1, f2)

    groups = {}
    for fi in range(n_faces):
        root = find(fi)
        groups.setdefault(root, []).append(fi)

    merged_faces = []
    for root, face_ids in groups.items():
        normals = face_normals[face_ids]
        avg_normal = normals.mean(axis=0)
        avg_normal = avg_normal / (np.linalg.norm(avg_normal) + 1e-12)

        # 收集该组所有三角形的顶点，去重后作为该宏观面的边界点集
        vert_ids = set()
        for fi in face_ids:
            for vid in hull.simplices[fi]:
                vert_ids.add(vid)
        vert_ids = list(vert_ids)
        verts_3d = vertices[hull.vertices][np.isin(hull.vertices, vert_ids)] \
            if False else vertices[vert_ids]  # 直接用全局顶点索引取坐标（hull.points按原始vertices索引）

        plane_point = verts_3d.mean(axis=0)

        # 将这些点投影到该平面的2D局部坐标系，求2D凸包，得到有序边界（避免内部点干扰立体角计算）
        # 构造平面内的两个正交基
        tmp = np.array([1.0, 0, 0]) if abs(avg_normal[0]) < 0.9 else np.array([0, 1.0, 0])
        basis_u = np.cross(avg_normal, tmp)
        basis_u /= np.linalg.norm(basis_u) + 1e-12
        basis_v = np.cross(avg_normal, basis_u)

        local_2d = np.column_stack([
            (verts_3d - plane_point) @ basis_u,
            (verts_3d - plane_point) @ basis_v
        ])

        if len(local_2d) >= 3:
            try:
                hull_2d = ConvexHull(local_2d)
                ordered_local = local_2d[hull_2d.vertices]
                ordered_3d = plane_point + np.outer(ordered_local[:, 0], basis_u) \
                             + np.outer(ordered_local[:, 1], basis_v)
            except Exception:
                ordered_3d = verts_3d
        else:
            ordered_3d = verts_3d

        merged_faces.append({
            "normal": avg_normal,
            "vertices_3d": ordered_3d,
            "plane_point": plane_point,
            "triangle_ids": face_ids,
        })

    return merged_faces


def _rotation_aligning_to_zup(normal: np.ndarray) -> np.ndarray:
    """
    构造一个旋转矩阵 R，使得 R @ (-normal) = [0,0,1]（即支撑面法向取反后对齐到+Z轴，
    物理含义：该支撑面朝下接触地面时，物体在此姿态下重新定向后的姿态）。

    用 Rodrigues 公式从"任意向量a旋转到目标向量b"构造旋转矩阵。
    """
    a = -normal / (np.linalg.norm(normal) + 1e-12)  # 支撑面法向取反 = 朝向地面方向
    b = np.array([0.0, 0.0, -1.0])  # 目标：让该方向对齐世界坐标系的-Z（即"朝下"）

    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        else:
            # a和b正好相反，绕任意垂直轴转180度
            perp = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
            axis = np.cross(a, perp)
            axis /= np.linalg.norm(axis) + 1e-12
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            return np.eye(3) + 2 * (K @ K)

    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    R = np.eye(3) + K + K @ K * ((1 - c) / (s ** 2))
    return R


def analyze_stable_poses(mesh, top_k=4, merge_angle_thresh_deg=3.0, verbose=True):
    """
    对输入的 trimesh.Trimesh 对象执行论文 3.2.1 节的稳定姿态分析。

    mesh: trimesh.Trimesh，顶点坐标已统一为米
    top_k: 论文式(1)中的K*大小，默认4
    返回: list[StablePoseResult]，按概率降序排列，长度<=top_k
    """
    vertices = mesh.vertices
    centroid = mesh.center_mass if hasattr(mesh, "center_mass") else vertices.mean(axis=0)
    # trimesh 的 center_mass 需要watertight网格才准确；若网格不是水密的，回退到顶点几何中心
    if not mesh.is_watertight:
        centroid = vertices.mean(axis=0)
        if verbose:
            print("  [analyze_stable_poses] ⚠ 网格非水密(non-watertight)，质心估计回退为顶点几何"
                  "平均值（而非真实体积质心），如需精确质心建议先对模型做补洞处理。")
    else:
        centroid = np.asarray(centroid)

    hull = ConvexHull(vertices)
    if verbose:
        print(f"  [analyze_stable_poses] 凸包顶点数={len(hull.vertices)}, "
              f"三角面数={len(hull.simplices)}")

    merged_faces = _merge_coplanar_faces(hull, vertices, angle_thresh_deg=merge_angle_thresh_deg)
    if verbose:
        print(f"  [analyze_stable_poses] 合并共面三角形后，得到 {len(merged_faces)} 个宏观支撑面")

    results_raw = []
    for fid, face in enumerate(merged_faces):
        normal = face["normal"]
        poly_verts = face["vertices_3d"]
        if len(poly_verts) < 3:
            continue

        # 质心投影到支撑面所在平面
        plane_point = face["plane_point"]
        d_to_plane = np.dot(centroid - plane_point, normal)
        projected_centroid = centroid - d_to_plane * normal

        # 物理筛选：只有当投影质心落在支撑面凸包内部（或附近）时，该姿态才是力学上可行的
        # （若质心投影落在支撑面之外，物体在该面朝下时会直接翻倒，不能算稳定支撑姿态）
        # 简化判定：用2D局部坐标系下的点在多边形内测试
        is_inside = _point_in_polygon_3d(projected_centroid, poly_verts, normal)

        W_k = polygon_solid_angle_from_apex(centroid, poly_verts)

        results_raw.append({
            "face_id": fid,
            "normal": normal,
            "plane_point": plane_point,
            "poly_verts": poly_verts,
            "solid_angle": W_k,
            "centroid_inside": is_inside,
        })

    # 论文式(1)的物理含义要求支撑面必须能让质心投影落在支撑范围内才稳定，
    # 这是论文未明确写出但隐含的力学约束（否则任意一个面都可以算"稳定"，
    # 与"翻倒所需角度"的物理直觉矛盾）。本实现优先选择 centroid_inside=True 的面，
    # 若数量不足top_k，再从剩余面中按立体角补足（保证一定能返回top_k个候选）。
    inside_faces = [r for r in results_raw if r["centroid_inside"]]
    outside_faces = [r for r in results_raw if not r["centroid_inside"]]

    inside_faces.sort(key=lambda r: r["solid_angle"], reverse=True)
    outside_faces.sort(key=lambda r: r["solid_angle"], reverse=True)

    selected_raw = inside_faces[:top_k]
    if len(selected_raw) < top_k:
        if verbose:
            print(f"  [analyze_stable_poses] 质心投影在支撑面内的面只有{len(inside_faces)}个，"
                  f"不足top_k={top_k}，从其余面中按立体角补足。")
        selected_raw += outside_faces[: top_k - len(selected_raw)]

    total_W = sum(r["solid_angle"] for r in selected_raw)
    if total_W <= 0:
        raise RuntimeError("所有候选支撑面的立体角总和为0，可能凸包构造失败")

    results = []
    for r in selected_raw:
        P_k = r["solid_angle"] / total_W
        R_zup = _rotation_aligning_to_zup(r["normal"])
        results.append(StablePoseResult(
            face_id=r["face_id"],
            normal=r["normal"],
            plane_point=r["plane_point"],
            polygon_vertices_3d=r["poly_verts"],
            solid_angle=r["solid_angle"],
            probability=P_k,
            rotation_to_zup=R_zup,
        ))

    if verbose:
        print(f"  [analyze_stable_poses] Top-{len(results)} 稳定姿态:")
        for r in results:
            print(f"    face_id={r.face_id}: W_k={r.solid_angle:.4f} sr, "
                  f"P_k={r.probability*100:.2f}%, normal={r.normal}")
        cum_prob = sum(r.probability for r in results)
        print(f"    累计概率(归一化后必为100%，此top_k子集内): {cum_prob*100:.2f}%")

    return results


def _point_in_polygon_3d(point_3d, poly_verts_3d, normal, tol=1e-6):
    """将3D点和3D多边形投影到该平面的2D局部坐标系，做2D点在多边形内测试（射线法）"""
    plane_point = poly_verts_3d.mean(axis=0)
    tmp = np.array([1.0, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1.0, 0])
    basis_u = np.cross(normal, tmp)
    basis_u /= np.linalg.norm(basis_u) + 1e-12
    basis_v = np.cross(normal, basis_u)

    p2d = np.array([(point_3d - plane_point) @ basis_u, (point_3d - plane_point) @ basis_v])
    poly2d = np.column_stack([
        (poly_verts_3d - plane_point) @ basis_u,
        (poly_verts_3d - plane_point) @ basis_v
    ])

    # 标准射线投射法 (ray casting point-in-polygon)
    n = len(poly2d)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly2d[i]
        xj, yj = poly2d[j]
        if ((yi > p2d[1]) != (yj > p2d[1])) and \
           (p2d[0] < (xj - xi) * (p2d[1] - yi) / (yj - yi + 1e-15) + xi):
            inside = not inside
        j = i
    return inside

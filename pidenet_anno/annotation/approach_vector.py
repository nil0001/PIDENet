# -*- coding: utf-8 -*-
"""
annotation/approach_vector.py
================================
论文 3.2.4 节 "Approach Vector" 的实现。

算法流程：
  1. 将2D抓取点对(p_k1, p_k2)（mask像素坐标，旋转后坐标系下）反投影到3D
     —— 这里我们处于"离线CAD标注"场景，不是真实深度图，所以反投影方式是：
        利用 projection.py 记录的 hit_points_3d（射线投影命中的真实3D表面点），
        通过最近邻匹配，找到与每个2D像素坐标对应的3D表面坐标
  2. 定义 Y轴 = 由p_km指向p_k2方向 (论文: "Y-axis is defined as the vector pointing
     from the centroid p_km to the second grasp point p_k2")
  3. 在p_km周围用KNN取局部邻域点，PCA估计Z轴（approach方向／法向）
  4. X轴 = Y × Z (右手系，论文Eq.3)
  5. 将上述坐标系从"投影旋转后坐标系"变换回原始物体坐标系

【关键修正说明】论文用"Y轴"指代由p_k1->p_k2方向的闭合轴，但在论文图2(b)和3.1节中，
approach vector被标记为v(对应z轴)，orientation vector被标记为u。为避免变量名混淆，
本模块内部使用论文3.2.4节的字母(x,y,z)做几何推导，最终输出时按论文3.1节的物理含义
转换为 (v=approach=z轴, u=orientation=y轴) 的命名，与本项目其余模块(grasp_scoring.py,
transform.py)保持接口一致。
"""

import numpy as np
from scipy.spatial import cKDTree


class GraspGeometry3D:
    """单个抓取候选的完整3D几何信息"""
    def __init__(self, p1_3d, p2_3d, center_3d, v_approach, u_orientation,
                 normal1, normal2, width):
        self.p1_3d = p1_3d            # 接触点1 (3,), 物体坐标系(米)
        self.p2_3d = p2_3d            # 接触点2 (3,)
        self.center = center_3d       # 抓取中心 p_km = (p1+p2)/2
        self.v = v_approach            # approach vector (单位向量, 论文z轴)
        self.u = u_orientation          # orientation vector (单位向量, 论文y轴/闭合轴方向)
        self.normal1 = normal1          # 接触点1处的局部表面法向(outward, 单位向量)
        self.normal2 = normal2          # 接触点2处的局部表面法向(outward, 单位向量)
        self.width = width               # w_k = ||p2-p1||  (米)

    def __repr__(self):
        return (f"GraspGeometry3D(center={self.center}, v={self.v}, u={self.u}, "
                f"w={self.width*1000:.1f}mm)")


def _pixel_to_3d_nearest(pixel_xy, hit_points_3d_xy, hit_points_3d, kdtree, max_dist_px):
    """
    给定一个2D像素坐标(在旋转后坐标系的物理xy下)，从射线投影记录的3D命中点中，
    找到xy平面投影距离最近的那个点，作为该2D坐标对应的3D表面坐标。

    pixel_xy: (2,) 物理坐标(米)，已经从像素索引转换为物理xy（不是行列索引）
    hit_points_3d_xy: (N,2) 所有命中点的xy分量
    hit_points_3d: (N,3) 所有命中点完整3D坐标
    kdtree: 对hit_points_3d_xy构建的cKDTree，避免重复构建提升效率
    max_dist_px实际是max_dist_m（沿用命名是历史遗留，下方统一用米）
    """
    dist, idx = kdtree.query(pixel_xy)
    if dist > max_dist_px:
        return None
    return hit_points_3d[idx]


def backproject_grasp_pairs_to_3d(pairs, projection_result, pixel_size, origin_xy,
                                     max_match_dist_factor=3.0, verbose=True):
    """
    将efd_contour.py输出的2D ContourPair列表，反投影为3D坐标对列表。

    pairs: list[ContourPair]，p1/p2是(row,col)像素坐标
    projection_result: ProjectionResult对象（来自projection.py），含hit_points_3d
    pixel_size, origin_xy: 来自projection.py，像素->物理坐标转换参数

    返回: list of dict {"p1_3d", "p2_3d", "source", ...}（仍在"旋转后坐标系"下，
          尚未变换回原始物体坐标系——该步骤在调用方完成，因为需要R_pose的逆变换）
    """
    hit_points_3d = projection_result.hit_points_3d
    hit_points_3d_xy = hit_points_3d[:, :2]
    kdtree = cKDTree(hit_points_3d_xy)
    max_match_dist = pixel_size * max_match_dist_factor

    results_3d = []
    skipped = 0

    for pair in pairs:
        # ContourPair.p1/p2 是 (row, col) 还是 (col, row)？
        # efd_contour.py 内部走 OpenCV 轮廓，OpenCV惯例坐标是 (col, row) = (x_pixel, y_pixel)
        # 这里统一约定: pair.p1[0]=col(x方向像素索引), pair.p1[1]=row(y方向像素索引)
        col1, row1 = pair.p1
        col2, row2 = pair.p2

        x1, y1 = origin_xy[0] + col1 * pixel_size, origin_xy[1] + row1 * pixel_size
        x2, y2 = origin_xy[0] + col2 * pixel_size, origin_xy[1] + row2 * pixel_size

        p1_3d = _pixel_to_3d_nearest(np.array([x1, y1]), hit_points_3d_xy, hit_points_3d,
                                       kdtree, max_match_dist)
        p2_3d = _pixel_to_3d_nearest(np.array([x2, y2]), hit_points_3d_xy, hit_points_3d,
                                       kdtree, max_match_dist)

        if p1_3d is None or p2_3d is None:
            skipped += 1
            continue

        results_3d.append({
            "p1_3d": p1_3d, "p2_3d": p2_3d, "source": pair.source,
            "defect_depth_px": pair.defect_depth, "hole_id": pair.hole_id,
        })

    if verbose:
        print(f"    [backproject_grasp_pairs_to_3d] 反投影成功{len(results_3d)}对, "
              f"跳过{skipped}对(2D点距最近3D命中点过远，可能落在投影边缘外)")

    return results_3d


def estimate_local_normal_pca(point_3d, mesh, k_neighbors=15, kdtree_mesh_vertices=None,
                                 degeneracy_eigval_ratio_thresh=0.15):
    """
    论文3.2.4节: "Extract a local neighborhood of N points around p_km using KNN search.
    The Z-axis is then estimated by performing PCA on these neighborhood points."

    point_3d: (3,) 查询点（旋转后坐标系下，物体表面附近）
    mesh: trimesh.Trimesh（已旋转到投影姿态的mesh）。需要 .vertices 以及
          .face_normals / .triangles_center（用于下方说明的主要估计方法）。
    k_neighbors: 兼容参数，仅在mesh缺少面信息、被迫回退到纯顶点PCA时才会用到
    kdtree_mesh_vertices: 兼容参数，同上仅在纯PCA回退路径中使用

    返回: (3,) 单位法向向量，已修正为outward方向

    【开发过程中两轮诊断后的最终方案，记录退化检测失败的真实教训】
    第一轮尝试：严格遵循论文字面描述，用顶点KNN+PCA估计法向。用一个32-sections的
    简化测试圆柱（细颈结构）实测发现：当查询点位于细长曲面上、邻域恰好采样到"绕一圈"
    的同一条周向环时，PCA给出的"法向"系统性偏离真实表面法向(72~88度量级误差)——
    环上的点近似共面，但这个"环平面"的法向恰好对齐了圆柱轴向，而非真实的径向法向。

    第二轮尝试：试图用PCA最小/最大特征值比例检测这种退化（猜想：退化时邻域应接近
    一条直线，最小特征值应显著偏小）。但实测发现该比例在所有测试k值下都远小于0.15
    （健康范围0.00003~0.037），说明"共面但选错平面"这种退化，在特征值比例上和正常的
    "贴合真实切平面"是无法区分的——共圆点集本身就会产生一个数值上"健康"但物理上错误
    的协方差矩阵。这说明基于特征值比例的检测思路从根本上不适用于这类退化模式。

    最终方案：彻底改变策略，不再以顶点PCA为默认方法，而是直接使用网格拓扑本身
    定义的面法向（face normal）作为主要估计依据——查询最近的若干个三角面，
    用距离加权平均其面法向。面法向由三角面的三个顶点的几何排列直接确定，不依赖
    任何"邻域点是否分布合理"的假设，因此不会受邻域采样模式（环状/线状/簇状）的影响，
    其精度只受限于网格本身的三角化分辨率——这是任何表面法向估计方法都无法绕开的
    基本物理限制，但至少不会引入额外的方向性错误。
    顶点PCA仅作为mesh缺少face_normals/triangles_center属性时的保底回退路径保留。
    """
    if hasattr(mesh, "face_normals") and hasattr(mesh, "triangles_center") and \
       mesh.face_normals is not None and len(mesh.face_normals) > 0:
        face_centers = mesh.triangles_center
        face_normals = mesh.face_normals
        face_kdtree = cKDTree(face_centers)
        n_faces_query = min(8, len(face_centers))
        fd, fidx = face_kdtree.query(point_3d, k=n_faces_query)
        fd = np.maximum(fd, 1e-9)
        weights = 1.0 / fd
        weights /= weights.sum()
        normal = (face_normals[fidx] * weights[:, None]).sum(axis=0)
        normal = normal / (np.linalg.norm(normal) + 1e-12)
    else:
        # 保底回退：mesh对象不含面信息时，退回纯顶点PCA（精度无法保证，仅用于不会
        # 触发上述退化场景的简单几何，或调用方明确知晓该限制的场景）
        if kdtree_mesh_vertices is None:
            kdtree_mesh_vertices = cKDTree(mesh.vertices)
        k_actual = min(k_neighbors, len(mesh.vertices))
        dist, idx = kdtree_mesh_vertices.query(point_3d, k=k_actual)
        neighbor_pts = mesh.vertices[idx]
        centered = neighbor_pts - neighbor_pts.mean(axis=0)
        cov = centered.T @ centered / len(neighbor_pts)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]
        normal = normal / (np.linalg.norm(normal) + 1e-12)

    centroid = mesh.vertices.mean(axis=0)
    ref_outward = point_3d - centroid
    if np.dot(normal, ref_outward) < 0:
        normal = -normal

    return normal / (np.linalg.norm(normal) + 1e-12)


def build_grasp_geometry(p1_3d, p2_3d, mesh, k_neighbors=15, kdtree_mesh_vertices=None):
    """
    给定一对3D接触点，构建完整的抓取几何信息（论文3.2.4节 Eq.3的右手系构造）。

    返回: GraspGeometry3D 对象（坐标仍在"旋转后坐标系"下，调用方负责变换回物体坐标系）
    """
    p1_3d = np.asarray(p1_3d, dtype=np.float64)
    p2_3d = np.asarray(p2_3d, dtype=np.float64)
    center = (p1_3d + p2_3d) / 2.0
    width = float(np.linalg.norm(p2_3d - p1_3d))

    if width < 1e-9:
        raise ValueError("两个接触点几乎重合，无法构建有效抓取几何")

    # 论文: "Y-axis ... pointing from the centroid p_km to the second grasp point p_k2"
    y_axis = (p2_3d - center) / (np.linalg.norm(p2_3d - center) + 1e-12)

    # Z轴: 围绕p_km做KNN+PCA法向估计（论文原话用p_km，本实现严格遵循）
    z_axis = estimate_local_normal_pca(center, mesh, k_neighbors, kdtree_mesh_vertices)

    # 确保z_axis与y_axis不近似平行（否则叉乘退化），若退化则用p1_3d附近的法向作为替代估计
    if abs(np.dot(y_axis, z_axis)) > 0.98:
        z_axis = estimate_local_normal_pca(p1_3d, mesh, k_neighbors, kdtree_mesh_vertices)

    # Gram-Schmidt 将z_axis投影垂直于y_axis（论文Eq.3的叉乘本身已保证x垂直于y和z，
    # 但为了让z_axis本身也严格垂直于y_axis,确保最终(x,y,z)构成正交右手系，这里先做一次修正)
    z_axis = z_axis - np.dot(z_axis, y_axis) * y_axis
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-12)

    # 论文 Eq.(3): x = y × z
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-12)

    # 接触点局部法向（用于后续grasp_scoring.py的antipodal alignment打分，Eq.5）
    normal1 = estimate_local_normal_pca(p1_3d, mesh, k_neighbors, kdtree_mesh_vertices)
    normal2 = estimate_local_normal_pca(p2_3d, mesh, k_neighbors, kdtree_mesh_vertices)

    # 输出命名对应论文3.1节: v=approach vector(本推导中的z_axis), u=orientation vector(y_axis)
    return GraspGeometry3D(
        p1_3d=p1_3d, p2_3d=p2_3d, center_3d=center,
        v_approach=z_axis, u_orientation=y_axis,
        normal1=normal1, normal2=normal2, width=width,
    )


def rotate_geometry_back_to_object_frame(geom: GraspGeometry3D, R_pose: np.ndarray) -> GraspGeometry3D:
    """
    将一个在"投影旋转后坐标系"下计算出的GraspGeometry3D，旋转变换回原始物体坐标系。
    R_pose 是stable_pose.py提供的 rotation_to_zup（原始坐标系->投影坐标系），
    因此这里用其逆（即转置，因为是旋转矩阵）做反变换。
    """
    R_inv = R_pose.T  # 旋转矩阵的逆 = 转置
    return GraspGeometry3D(
        p1_3d=R_inv @ geom.p1_3d,
        p2_3d=R_inv @ geom.p2_3d,
        center_3d=R_inv @ geom.center,
        v_approach=R_inv @ geom.v,
        u_orientation=R_inv @ geom.u,
        normal1=R_inv @ geom.normal1,
        normal2=R_inv @ geom.normal2,
        width=geom.width,
    )

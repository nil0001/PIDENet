# -*- coding: utf-8 -*-
"""
annotation/projection.py
==========================
论文 3.2.2 节 "Ray-Casting Orthographic Projection" 的实现。

算法流程：
  1. 将物体按某个稳定姿态旋转到"该支撑面朝下"的标准位姿
  2. 在物体上方建立高分辨率像素网格，沿世界坐标系-Z方向发射密集射线
  3. 用 trimesh 的射线-网格相交检测，命中点生成二值mask
  4. 【薄壁容器特判】若物体在该姿态下，射线命中点中"位于bbox上方30%高度区域"
     的点集所形成的投影轮廓，相比"全部命中点"的投影轮廓有显著的空洞结构差异，
     则判定为薄壁中空容器，仅用上方30%区域的命中点重新生成mask（环形化处理）；
     否则（如LINEMOD的ape等实心物体）直接使用全部命中点的mask。

注意：论文里"动态高度阈值"机制的触发条件本身没有给出量化标准，这是本实现按
config.yaml 中 cavity_pixel_ratio_threshold 补充的判定逻辑（README中D2已说明）。
"""

import numpy as np
import trimesh


class ProjectionResult:
    def __init__(self, mask, pixel_size, origin_xy, hit_points_3d, is_thin_wall, R_pose):
        self.mask = mask                  # (H, W) bool ndarray, True=物体投影命中
        self.pixel_size = pixel_size       # 每个像素对应的物理尺寸（米）
        self.origin_xy = origin_xy         # mask[0,0]像素中心对应的物体局部坐标系(旋转后)下的(x,y)（米）
        self.hit_points_3d = hit_points_3d  # (N,3) 实际用于生成mask的命中点，旋转后坐标系下（米）
        self.is_thin_wall = is_thin_wall   # 是否触发了薄壁环形化处理
        self.R_pose = R_pose                # 本次投影使用的旋转矩阵（原始物体坐标系 -> 投影用旋转坐标系）

    def pixel_to_local_xy(self, row, col):
        """mask像素坐标 -> 旋转后坐标系下的物理(x,y)坐标（米）"""
        x = self.origin_xy[0] + col * self.pixel_size
        y = self.origin_xy[1] + row * self.pixel_size
        return x, y


def _rotate_mesh(mesh: trimesh.Trimesh, R: np.ndarray) -> trimesh.Trimesh:
    """返回一个顶点已被R旋转的新mesh副本（不修改原mesh）"""
    rotated = mesh.copy()
    rotated.vertices = (R @ mesh.vertices.T).T
    return rotated


def raycast_orthographic_projection(mesh: trimesh.Trimesh, R_pose: np.ndarray,
                                      resolution: int = 400,
                                      thin_wall_height_ratio: float = 0.30,
                                      cavity_pixel_ratio_threshold: float = 0.02,
                                      verbose=True) -> ProjectionResult:
    """
    对给定姿态(R_pose对齐后该姿态朝下)的物体执行射线投影。

    mesh: 原始物体坐标系下的网格（米）
    R_pose: 来自 stable_pose.py 的 rotation_to_zup，将该支撑面法向对齐到世界-Z方向
    resolution: 投影网格分辨率（正方形像素边长数）
    """
    rotated_mesh = _rotate_mesh(mesh, R_pose)
    bounds = rotated_mesh.bounds  # (2,3): [min_xyz, max_xyz]
    extent_xy = bounds[1, :2] - bounds[0, :2]
    pixel_size = max(extent_xy) / resolution * 1.05  # 留5%边距，避免边缘点落在网格外

    nx = int(np.ceil(extent_xy[0] / pixel_size)) + 2
    ny = int(np.ceil(extent_xy[1] / pixel_size)) + 2
    origin_x = bounds[0, 0] - pixel_size
    origin_y = bounds[0, 1] - pixel_size

    xs = origin_x + (np.arange(nx) + 0.5) * pixel_size
    ys = origin_y + (np.arange(ny) + 0.5) * pixel_size
    grid_x, grid_y = np.meshgrid(xs, ys)  # grid_x/grid_y shape: (ny, nx)

    z_top = bounds[1, 2] + 0.01  # 射线起点：物体最高点上方1cm
    ray_origins = np.stack([
        grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, z_top)
    ], axis=1)
    ray_directions = np.tile(np.array([0.0, 0.0, -1.0]), (len(ray_origins), 1))

    # trimesh的射线相交检测：返回命中的射线索引、命中点位置、命中的三角面索引
    # 使用 ray.intersects_location，对于不要求最近交点（multiple_hits=True时返回所有交点，
    # 这里我们只需要"沿z轴最上方的第一个交点"，因此用 intersects_first 更高效，但部分版本
    # trimesh对凸/凹网格的intersects_first实现可能不稳定，这里改用更鲁棒的多交点+取最大z的方式
    locations, ray_idx, tri_idx = rotated_mesh.ray.intersects_location(
        ray_origins, ray_directions, multiple_hits=True
    )

    if len(locations) == 0:
        raise RuntimeError("射线投影未命中任何三角面，请检查mesh是否为空或R_pose是否异常")

    # 对于命中多次的射线（射线穿过物体内部多个表面，如凹陷处），取z最大（最上方）的那个交点
    # 作为该像素的"可见表面高度"——这正是正交投影应该呈现的最上方可见点
    df_ray_idx = ray_idx
    df_z = locations[:, 2]
    order = np.lexsort((-df_z, df_ray_idx))  # 先按ray_idx分组，组内按z降序
    sorted_ray_idx = df_ray_idx[order]
    sorted_locations = locations[order]
    _, first_pos = np.unique(sorted_ray_idx, return_index=True)
    top_hit_ray_idx = sorted_ray_idx[first_pos]
    top_hit_locations = sorted_locations[first_pos]

    mask_full = np.zeros((ny, nx), dtype=bool)
    rows = top_hit_ray_idx // nx
    cols = top_hit_ray_idx % nx
    mask_full[rows, cols] = True

    if verbose:
        print(f"    [raycast] 网格={ny}x{nx}, pixel_size={pixel_size*1000:.3f}mm, "
              f"命中像素数={mask_full.sum()}/{mask_full.size} "
              f"({100*mask_full.sum()/mask_full.size:.1f}%)")

    # ---------------- 薄壁容器判定 (论文3.2.2节后半部分) ----------------
    z_min_obj, z_max_obj = bounds[0, 2], bounds[1, 2]
    height = z_max_obj - z_min_obj
    z_threshold = z_max_obj - thin_wall_height_ratio * height

    upper_mask_hit = top_hit_locations[:, 2] >= z_threshold
    upper_locations = top_hit_locations[upper_mask_hit]
    upper_ray_idx = top_hit_ray_idx[upper_mask_hit]

    mask_upper = np.zeros((ny, nx), dtype=bool)
    if len(upper_ray_idx) > 0:
        rows_u = upper_ray_idx // nx
        cols_u = upper_ray_idx % nx
        mask_upper[rows_u, cols_u] = True

    # 用"内部空洞像素占bbox投影面积的比例"衡量两个mask的拓扑差异：
    # 若上方30%区域生成的mask相比完整mask多出了显著的内部空洞（说明完整mask把空腔"实心化"了），
    # 则判定为薄壁容器，应该用 mask_upper 替代 mask_full
    def _internal_cavity_ratio(binary_mask):
        from scipy import ndimage
        if binary_mask.sum() == 0:
            return 0.0
        filled = ndimage.binary_fill_holes(binary_mask)
        cavity = filled & (~binary_mask)
        # 分母必须是"填充后的完整轮廓面积"(即物体外轮廓包络的总像素数,包含空洞)，
        # 而不是"实心像素数"本身——否则当空洞面积大于实心面积时，比例会超过1，
        # 物理意义上的"空洞占整体轮廓的比例"就失真了(此前的bug)。
        total_area = filled.sum()
        if total_area == 0:
            return 0.0
        return cavity.sum() / total_area

    cavity_full = _internal_cavity_ratio(mask_full)
    cavity_upper = _internal_cavity_ratio(mask_upper) if mask_upper.sum() > 0 else 0.0

    is_thin_wall = (cavity_upper - cavity_full) > cavity_pixel_ratio_threshold

    if verbose:
        print(f"    [raycast] 内部空洞比例: full={cavity_full:.4f}, upper30%={cavity_upper:.4f}, "
              f"差值={cavity_upper-cavity_full:.4f}, 薄壁判定阈值={cavity_pixel_ratio_threshold} "
              f"=> 判定为薄壁容器: {is_thin_wall}")

    if is_thin_wall and mask_upper.sum() > 0:
        final_mask = mask_upper
        final_hit_points = upper_locations
        if verbose:
            print(f"    [raycast] 触发薄壁环形化处理，使用上方{thin_wall_height_ratio*100:.0f}%"
                  f"区域命中点重建mask")
    else:
        final_mask = mask_full
        final_hit_points = top_hit_locations

    return ProjectionResult(
        mask=final_mask,
        pixel_size=pixel_size,
        origin_xy=(origin_x, origin_y),
        hit_points_3d=final_hit_points,
        is_thin_wall=is_thin_wall,
        R_pose=R_pose,
    )

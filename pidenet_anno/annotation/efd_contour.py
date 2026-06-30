# -*- coding: utf-8 -*-
"""
annotation/efd_contour.py
============================
论文 3.2.3 节 "EFD Smooth Curve Reconstruction" 的实现。

算法流程：
  1. 对二值mask做连通域分析，区分外轮廓 C_out 和内部孔洞轮廓 {C_in,k}
     （用OpenCV的层级轮廓检测 cv2.RETR_CCOMP/RETR_TREE 天然支持这一区分）
  2. 对每条轮廓，用 Elliptic Fourier Descriptors (EFD) 做平滑拟合，
     得到论文 Eq.(2) 描述的连续曲线 S(t)
  3. 双分支抓取点提取：
     (a) 外轮廓分支：计算凸包+凸性缺陷，取每个缺陷的最深凹点作为 p_k1，
         沿缺陷中点指向内部方向发射法线方向射线，找到对侧边界交点作为 p_k2
     (b) 内孔分支（仅当存在内孔时触发）：计算内外轮廓采样点间的全局最短跨轮廓连接

返回的抓取点对均为 mask 的2D像素坐标，后续由 approach_vector.py 反投影到3D。
"""

import numpy as np
import cv2
import pyefd


class ContourPair:
    """一对2D抓取接触点（像素坐标），附带来源信息"""
    def __init__(self, p1, p2, source="outer_defect", defect_depth=None, hole_id=None):
        self.p1 = np.array(p1, dtype=np.float64)  # (row, col) 像素坐标
        self.p2 = np.array(p2, dtype=np.float64)
        self.source = source            # "outer_defect" 或 "inner_hole"
        self.defect_depth = defect_depth  # 仅 outer_defect 有意义：凸性缺陷深度（像素）
        self.hole_id = hole_id            # 仅 inner_hole 有意义：对应哪个内孔

    def width_px(self):
        return float(np.linalg.norm(self.p2 - self.p1))

    def __repr__(self):
        return f"ContourPair(p1={self.p1}, p2={self.p2}, source={self.source})"


def extract_contours_hierarchy(mask: np.ndarray, min_area_px: int = 30, verbose=True):
    """
    用OpenCV层级轮廓检测，区分外轮廓与内部孔洞轮廓。

    返回: (outer_contour, inner_contours)
      outer_contour: (N,2) ndarray，[col,row]格式（OpenCV惯例），物体外边界
      inner_contours: list of (N,2) ndarray，每个是一个内部孔洞的边界
    """
    mask_u8 = (mask.astype(np.uint8)) * 255
    contours, hierarchy = cv2.findContours(mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        raise RuntimeError("mask中未检测到任何轮廓，可能mask为全0")

    # hierarchy[i] = [next, prev, first_child, parent]
    # parent == -1 表示该轮廓是顶层（外轮廓）；parent != -1 表示是某个外轮廓的孔洞
    outer_idx = None
    outer_area = -1
    inner_indices = []

    for i, h in enumerate(hierarchy[0]):
        area = cv2.contourArea(contours[i])
        if area < min_area_px:
            continue
        parent = h[3]
        if parent == -1:
            if area > outer_area:
                if outer_idx is not None:
                    # 之前的"最大外轮廓"如果不是真正的最大值，需要重新归类逻辑这里简化：
                    # 直接保留面积最大的顶层轮廓为outer，其余顶层轮廓视为噪声忽略
                    pass
                outer_area = area
                outer_idx = i
        else:
            inner_indices.append(i)

    if outer_idx is None:
        raise RuntimeError("未找到有效外轮廓（所有候选轮廓面积均小于min_area_px阈值）")

    outer_contour = contours[outer_idx].reshape(-1, 2).astype(np.float64)
    inner_contours = [contours[i].reshape(-1, 2).astype(np.float64) for i in inner_indices
                       if cv2.contourArea(contours[i]) >= min_area_px]

    if verbose:
        print(f"    [extract_contours_hierarchy] 外轮廓点数={len(outer_contour)}, "
              f"面积={outer_area:.1f}px², 检测到内孔数={len(inner_contours)}")
        for k, ic in enumerate(inner_contours):
            print(f"      内孔#{k}: 点数={len(ic)}, 面积={cv2.contourArea(ic.astype(np.int32)):.1f}px²")

    return outer_contour, inner_contours


def fit_efd_smooth_contour(contour_px: np.ndarray, harmonic_order: int = 10,
                             n_resample: int = 400):
    """
    对给定离散轮廓点序列做EFD拟合，返回重采样后的平滑闭合曲线点集。

    contour_px: (N,2) ndarray，[col,row]格式
    harmonic_order: 论文Eq.(2)中的k（谐波阶数）
    n_resample: 在拟合曲线上重新采样多少个点，用于后续凸包/缺陷分析

    返回: (n_resample, 2) ndarray，平滑后的轮廓点，[col,row]格式
    """
    coeffs = pyefd.elliptic_fourier_descriptors(contour_px, order=harmonic_order, normalize=False)
    locus = pyefd.calculate_dc_coefficients(contour_px)
    smooth_pts = pyefd.reconstruct_contour(coeffs, locus=locus, num_points=n_resample)
    return smooth_pts


def find_outer_defect_grasp_pairs(smooth_contour: np.ndarray, mask_shape, verbose=True):
    """
    论文3.2.3节"For the peripheral contour C_out"段落的实现：
      1. 计算平滑轮廓的凸包
      2. 计算凸性缺陷 (convexity defects)
      3. 每个缺陷的最深凹点 = p_k1
      4. 从缺陷起止点中点沿法线方向（指向物体内部）发射射线，找对侧边界交点 = p_k2

    smooth_contour: (N,2) ndarray，[col,row]格式，来自fit_efd_smooth_contour
    返回: list[ContourPair]
    """
    contour_int = smooth_contour.astype(np.int32).reshape(-1, 1, 2)
    hull_indices = cv2.convexHull(contour_int, returnPoints=False)

    if len(hull_indices) < 3:
        if verbose:
            print("    [find_outer_defect_grasp_pairs] 凸包退化（点数<3），无法计算缺陷")
        return []

    try:
        defects = cv2.convexityDefects(contour_int, hull_indices)
    except cv2.error as e:
        if verbose:
            print(f"    [find_outer_defect_grasp_pairs] convexityDefects计算失败: {e}")
        return []

    if defects is None:
        if verbose:
            print("    [find_outer_defect_grasp_pairs] 该轮廓本身接近凸形，无显著凹陷缺陷")
        return []

    pairs = []
    H, W = mask_shape

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start_pt = contour_int[s][0].astype(np.float64)   # 缺陷起点（凸包顶点）
        end_pt = contour_int[e][0].astype(np.float64)     # 缺陷终点（凸包顶点）
        far_pt = contour_int[f][0].astype(np.float64)      # 缺陷最深点（轮廓上离凸包边最远的点）
        depth_px = d / 256.0  # OpenCV的convexityDefects深度值按定点数*256编码

        if depth_px < 1.0:
            continue  # 过浅的缺陷视为噪声，跳过

        p_k1 = far_pt  # 最深凹点 = 主接触点

        mid_pt = (start_pt + end_pt) / 2.0
        # 法线方向：从mid_pt指向far_pt的方向，再延伸寻找对侧边界
        # （这与论文描述"along the midpoint of the convex hull defect pointing towards p_k2,
        #   a ray is traced in the normal direction towards the interior" 一致：
        #   以缺陷中点为起点，沿着"指向物体内部"的方向（即从凸包边界指向最深凹点再继续延伸）发射射线）
        direction = far_pt - mid_pt
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-6:
            continue
        direction = direction / dir_norm

        p_k2 = _ray_march_to_far_boundary(mid_pt, direction, smooth_contour, H, W)

        if p_k2 is not None:
            pairs.append(ContourPair(p_k1, p_k2, source="outer_defect", defect_depth=depth_px))

    if verbose:
        print(f"    [find_outer_defect_grasp_pairs] 检测到{defects.shape[0] if defects is not None else 0}个"
              f"凸性缺陷，有效生成{len(pairs)}对抓取候选点")

    return pairs


def _ray_march_to_far_boundary(start_pt, direction, smooth_contour, H, W, max_steps=2000, step_size=0.5):
    """
    从 start_pt 沿 direction 方向步进，直到穿出轮廓（即与平滑轮廓多边形相交于对侧边界）。

    用 cv2.pointPolygonTest 逐步判断点是否仍在轮廓内，找到"从内部刚好穿出到外部"的临界点，
    再回退到轮廓上最近的交点作为p_k2。这是对论文"a ray is traced...until it penetrates and
    touches the opposite edge"的直接实现。
    """
    contour_for_test = smooth_contour.astype(np.float32).reshape(-1, 1, 2)

    prev_pt = start_pt.copy()
    was_inside = cv2.pointPolygonTest(contour_for_test, tuple(prev_pt), False) >= 0

    for step in range(1, max_steps):
        cur_pt = start_pt + direction * step * step_size
        if cur_pt[0] < 0 or cur_pt[0] >= W or cur_pt[1] < 0 or cur_pt[1] >= H:
            return None  # 射线跑出图像边界，未找到对侧交点

        is_inside = cv2.pointPolygonTest(contour_for_test, tuple(cur_pt), False) >= 0

        if was_inside and not is_inside:
            # 刚好穿出轮廓，用二分法精细定位边界交点
            lo_pt, hi_pt = prev_pt.copy(), cur_pt.copy()
            for _ in range(20):
                mid = (lo_pt + hi_pt) / 2.0
                mid_inside = cv2.pointPolygonTest(contour_for_test, tuple(mid), False) >= 0
                if mid_inside:
                    lo_pt = mid
                else:
                    hi_pt = mid
            return (lo_pt + hi_pt) / 2.0

        prev_pt = cur_pt
        was_inside = is_inside

    return None  # 步进次数耗尽仍未找到边界


def find_inner_hole_grasp_pair(outer_contour: np.ndarray, inner_contour: np.ndarray,
                                  n_sample=100, verbose=True):
    """
    论文3.2.3节"For the internal handle hole C_in"段落的实现：
    计算内孔轮廓与外轮廓采样点之间的全局最短跨轮廓连接。

    (a*, b*) = argmin_{a,b} ||p_{a,in} - p_{b,out}||

    这定位的是"手柄壁最薄、最易被夹爪环抱"的横截面位置。

    outer_contour, inner_contour: (N,2) ndarray，[col,row]格式
    返回: ContourPair 或 None（若计算失败）
    """
    if len(outer_contour) == 0 or len(inner_contour) == 0:
        return None

    idx_out = np.linspace(0, len(outer_contour) - 1, n_sample).astype(int)
    idx_in = np.linspace(0, len(inner_contour) - 1, n_sample).astype(int)
    sampled_out = outer_contour[idx_out]
    sampled_in = inner_contour[idx_in]

    # 计算所有内孔采样点 到 所有外轮廓采样点 的欧氏距离矩阵
    diff = sampled_in[:, None, :] - sampled_out[None, :, :]  # (n_sample, n_sample, 2)
    dist_matrix = np.linalg.norm(diff, axis=2)

    a_star, b_star = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
    p_a_in = sampled_in[a_star]
    p_b_out = sampled_out[b_star]
    min_dist = dist_matrix[a_star, b_star]

    if verbose:
        print(f"    [find_inner_hole_grasp_pair] 最短跨轮廓连接: 内孔点{p_a_in} <-> "
              f"外轮廓点{p_b_out}, 距离={min_dist:.2f}px")

    return ContourPair(p_a_in, p_b_out, source="inner_hole")


def extract_all_grasp_pairs(mask: np.ndarray, efd_cfg: dict, verbose=True):
    """
    顶层封装函数：给定二值mask，依次执行连通域分析+EFD拟合+双分支抓取点提取。

    efd_cfg: 来自config.yaml的efd配置段
    返回: list[ContourPair]
    """
    outer_raw, inner_raw_list = extract_contours_hierarchy(
        mask, min_area_px=efd_cfg["min_contour_area_px"], verbose=verbose
    )

    smooth_outer = fit_efd_smooth_contour(
        outer_raw, harmonic_order=efd_cfg["harmonic_order"],
        n_resample=efd_cfg["contour_resample_points"]
    )

    pairs = find_outer_defect_grasp_pairs(smooth_outer, mask.shape, verbose=verbose)

    for hole_id, inner_raw in enumerate(inner_raw_list):
        smooth_inner = fit_efd_smooth_contour(
            inner_raw, harmonic_order=efd_cfg["harmonic_order"],
            n_resample=max(50, efd_cfg["contour_resample_points"] // 2)
        )
        pair = find_inner_hole_grasp_pair(smooth_outer, smooth_inner, verbose=verbose)
        if pair is not None:
            pair.hole_id = hole_id
            pairs.append(pair)

    if verbose:
        print(f"    [extract_all_grasp_pairs] 总计提取 {len(pairs)} 对候选抓取点 "
              f"(外轮廓缺陷={sum(1 for p in pairs if p.source=='outer_defect')}, "
              f"内孔配对={sum(1 for p in pairs if p.source=='inner_hole')})")

    return pairs, smooth_outer

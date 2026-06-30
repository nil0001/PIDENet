# -*- coding: utf-8 -*-
"""
visualize_3d.py
=================
对 annotation/pipeline.py 生成的物体坐标系候选抓取列表做3D可视化。

要求（用户明确指定）：可视化时必须包含物体的CAD模型本身，而不只是孤立的抓取点/箭头。

实现两套输出：
  1. open3d交互式窗口（可旋转/缩放查看，运行时弹出）
  2. matplotlib静态PNG（自动保存到output/，便于无GUI环境下查看结果或写入报告）

每个候选抓取用以下元素可视化：
  - 抓取中心：小球标记
  - approach vector (v)：蓝色箭头
  - orientation/closing axis (u)：红色箭头
  - 两个接触点 p1, p2：黄色小球
  - 连接p1-p2的线段：表示夹爪闭合方向跨度
"""

import numpy as np
import os

try:
    import open3d as o3d
except ImportError:
    o3d = None

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def _candidate_color(score_Q, feasible):
    """根据Q分数和可行性返回一个RGB颜色，用于区分候选质量"""
    if not feasible:
        return (0.6, 0.6, 0.6)  # 灰色：不可行
    # 可行候选按Q值在红(低分)->绿(高分)之间插值
    q = np.clip(score_Q, 0.0, 1.0)
    return (1.0 - q, q, 0.1)


def visualize_candidates_open3d(mesh, candidates, top_n=10, arrow_length=0.02,
                                   save_path=None, show_window=True):
    """
    用open3d渲染CAD模型+候选抓取的3D交互式可视化。

    mesh: trimesh.Trimesh，物体坐标系下（米单位）
    candidates: list[GraspCandidate]，来自pipeline.generate_object_frame_candidates
                (假设已按Q降序排列，此函数只取前top_n个可视化，避免画面过于杂乱)
    arrow_length: 箭头可视化长度（米），按物体尺寸自适应更合理，调用方可按需调整
    save_path: 若提供，会用open3d离屏渲染保存一张截图到该路径
    show_window: 是否弹出交互式窗口（无GUI环境下应设为False，仅依赖save_path输出）
    """
    if o3d is None:
        raise ImportError("需要open3d库：pip install open3d")

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.paint_uniform_color([0.75, 0.75, 0.78])

    geometries = [o3d_mesh]

    # 物体坐标系参考系（小尺寸，帮助辨认朝向）
    obj_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=arrow_length * 0.8, origin=[0, 0, 0]
    )
    geometries.append(obj_frame)

    selected = candidates[:top_n]

    for cand in selected:
        geom = cand.geom
        color = _candidate_color(cand.score.Q, cand.score.feasible)

        center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=arrow_length * 0.12)
        center_sphere.translate(geom.center)
        center_sphere.paint_uniform_color(color)
        geometries.append(center_sphere)

        for contact_pt in [geom.p1_3d, geom.p2_3d]:
            contact_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=arrow_length * 0.08)
            contact_sphere.translate(contact_pt)
            contact_sphere.paint_uniform_color([1.0, 0.85, 0.0])
            geometries.append(contact_sphere)

        line_p1p2 = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([geom.p1_3d, geom.p2_3d]),
            lines=o3d.utility.Vector2iVector([[0, 1]])
        )
        line_p1p2.colors = o3d.utility.Vector3dVector([[1.0, 0.85, 0.0]])
        geometries.append(line_p1p2)

        v_arrow = _make_arrow(geom.center, geom.v, length=arrow_length, color=[0.1, 0.3, 0.95])
        geometries.append(v_arrow)

        u_arrow = _make_arrow(geom.center, geom.u, length=arrow_length, color=[0.9, 0.1, 0.15])
        geometries.append(u_arrow)

    if show_window:
        o3d.visualization.draw_geometries(
            geometries,
            window_name="PIDENet物体坐标系候选抓取可视化 (蓝=approach v, 红=orientation u, 黄=接触点)",
            width=1200, height=900,
        )

    if save_path is not None:
        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=1600, height=1200)
            for g in geometries:
                vis.add_geometry(g)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(save_path)
            vis.destroy_window()

            # 离屏渲染在某些软件渲染后端(如缺少硬件加速的OSMesa环境)下可能"成功"返回但
            # 实际写出全黑/空白图像——这是开发过程中在沙箱容器环境中发现的真实兼容性问题，
            # 并非本代码的逻辑错误。这里做一个简单的健全性检查：若生成的图像几乎全黑
            # （所有像素均值接近0），提示用户改用show_window=True获取交互式窗口结果，
            # 或在自己的桌面环境（通常有真实显卡/OpenGL驱动）下重新运行截图功能。
            import cv2 as _cv2
            check_img = _cv2.imread(save_path)
            if check_img is not None and check_img.mean() < 1.0:
                print(f"    ⚠ 警告：截图 {save_path} 渲染结果几乎全黑，这通常是无GPU加速的"
                      f"离屏渲染环境(如某些Linux容器/远程服务器)的已知兼容性问题，不代表"
                      f"可视化逻辑有误。建议改用 show_window=True 在本机弹出交互式窗口查看，"
                      f"或参考 visualize_candidates_matplotlib() 获取保证可用的静态图替代方案。")
        except Exception as e:
            print(f"    ⚠ open3d离屏截图失败: {e}")
            print(f"    建议改用 show_window=True 弹出交互式窗口，或使用 "
                  f"visualize_candidates_matplotlib() 作为不依赖GPU渲染的备选方案。")

    return geometries


def _make_arrow(origin, direction, length=0.02, color=(0, 0, 1), shaft_radius_ratio=0.06):
    """构造一个open3d箭头几何体，从origin沿direction方向延伸length长度"""
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    cylinder_height = length * 0.7
    cone_height = length * 0.3
    cylinder_radius = length * shaft_radius_ratio
    cone_radius = cylinder_radius * 1.8

    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=cylinder_radius, cone_radius=cone_radius,
        cylinder_height=cylinder_height, cone_height=cone_height,
    )
    # create_arrow默认沿+z方向生成，需要旋转到指向direction
    z_axis = np.array([0, 0, 1.0])
    rot_axis = np.cross(z_axis, direction)
    rot_axis_norm = np.linalg.norm(rot_axis)

    if rot_axis_norm < 1e-8:
        if np.dot(z_axis, direction) < 0:
            R = o3d.geometry.get_rotation_matrix_from_axis_angle([np.pi, 0, 0])
        else:
            R = np.eye(3)
    else:
        rot_axis = rot_axis / rot_axis_norm
        angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis * angle)

    arrow.rotate(R, center=[0, 0, 0])
    arrow.translate(origin)
    arrow.paint_uniform_color(color)
    return arrow


def visualize_candidates_matplotlib(mesh, candidates, top_n=10, arrow_length=0.02,
                                       save_path="output/object_grasp_candidates_3d.png"):
    """
    用matplotlib渲染静态3D图（无需GUI/显示器，适合任何环境保存结果截图）。
    可视化元素含义与open3d版本一致。
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    verts = mesh.vertices
    faces = mesh.faces
    mesh_collection = Line3DCollection(
        [verts[face] for face in faces[::max(1, len(faces) // 3000)]],  # 降采样避免过密
        colors=(0.5, 0.5, 0.55, 0.25), linewidths=0.3
    )
    ax.add_collection3d(mesh_collection)
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], s=0.5, c="gray", alpha=0.15)

    selected = candidates[:top_n]
    for cand in selected:
        geom = cand.geom
        color = _candidate_color(cand.score.Q, cand.score.feasible)

        ax.scatter(*geom.center, color=color, s=60, marker="o", edgecolors="black", linewidths=0.5)

        p1, p2 = geom.p1_3d, geom.p2_3d
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color="gold", linewidth=2, alpha=0.9)
        ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color="gold", s=30, marker="^")

        v_end = geom.center + geom.v * arrow_length
        ax.plot([geom.center[0], v_end[0]], [geom.center[1], v_end[1]],
                [geom.center[2], v_end[2]], color="blue", linewidth=2.5,
                label="approach (v)" if cand is selected[0] else None)

        u_end = geom.center + geom.u * arrow_length
        ax.plot([geom.center[0], u_end[0]], [geom.center[1], u_end[1]],
                [geom.center[2], u_end[2]], color="red", linewidth=2.5,
                label="orientation (u)" if cand is selected[0] else None)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Object-Frame Grasp Candidates (Top-{len(selected)}, "
                 f"green=high-score feasible, red=low-score feasible, gray=infeasible)")
    ax.legend(loc="upper right")

    max_range = np.array([verts[:, i].max() - verts[:, i].min() for i in range(3)]).max() / 2.0
    mid = verts.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize_candidates_matplotlib] 已保存静态图: {save_path}")
    return save_path

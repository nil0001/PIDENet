# -*- coding: utf-8 -*-
"""
test_collision_pruning.py — 验证 collision_pruning.py 的针孔投影与碰撞代理打分逻辑。
"""
import sys
sys.path.insert(0, ".")
import numpy as np
from annotation.collision_pruning import (
    project_points_to_depth_pixels, compute_collision_proxy_score, CollisionVolumes
)


def test_pinhole_projection_basic():
    """验证针孔投影公式: 光轴上的点应投影到主点(cx,cy)，偏移点按 col=x*fx/z+cx 计算"""
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)

    pt_center = np.array([[0, 0, 1.0]])
    pixel_xy, depth, valid = project_points_to_depth_pixels(pt_center, K)
    assert tuple(pixel_xy[0]) == (320, 240), f"光轴投影错误: {pixel_xy[0]}"
    assert abs(depth[0] - 1.0) < 1e-9

    pt_offset = np.array([[0.1, 0, 1.0]])
    pixel_xy2, _, _ = project_points_to_depth_pixels(pt_offset, K)
    assert tuple(pixel_xy2[0]) == (370, 240), f"偏移点投影错误: {pixel_xy2[0]}"

    print("[test_pinhole_projection_basic] PASS")


def test_collision_score_no_occlusion():
    """场景深度远大于抓取点深度 => 无遮挡 => P_coll应接近1.0"""
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    H, W = 480, 640
    depth_image = np.full((H, W), 2.0)  # 场景深度2米，远超抓取点约1米

    grasp_cam = {"center": np.array([0., 0., 1.]), "v": np.array([0., 0., 1.]),
                 "u": np.array([1., 0., 0.]), "w": 0.06}
    gripper_cfg = {"finger_thickness": 0.008, "finger_length": 0.045, "approach_clearance": 0.06}
    collision_cfg = {"alpha": 3.0, "depth_occlusion_margin": 0.005}

    P = compute_collision_proxy_score(grasp_cam, depth_image, K, gripper_cfg, collision_cfg)
    assert P > 0.99, f"无遮挡场景P_coll应接近1.0，实际={P}"
    print(f"[test_collision_score_no_occlusion] P_coll={P:.4f}, PASS")


def test_collision_score_full_occlusion():
    """场景深度远小于抓取点深度（遮挡物挡在前方） => P_coll应显著低"""
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    H, W = 480, 640
    depth_image = np.full((H, W), 0.5)  # 遮挡物在0.5米处，远比抓取点(约1米)更近

    grasp_cam = {"center": np.array([0., 0., 1.]), "v": np.array([0., 0., 1.]),
                 "u": np.array([1., 0., 0.]), "w": 0.06}
    gripper_cfg = {"finger_thickness": 0.008, "finger_length": 0.045, "approach_clearance": 0.06}
    collision_cfg = {"alpha": 3.0, "depth_occlusion_margin": 0.005}

    P = compute_collision_proxy_score(grasp_cam, depth_image, K, gripper_cfg, collision_cfg)
    assert P < 0.1, f"完全遮挡场景P_coll应显著低，实际={P}"
    print(f"[test_collision_score_full_occlusion] P_coll={P:.4f}, PASS")


def test_margin_tolerance_with_high_res_camera():
    """
    【关键回归测试，记录一次真实诊断过程】
    用高分辨率虚拟相机(避免多点共享同一像素导致的离散化伪影)，验证margin容差逻辑
    在像素不重叠条件下的数值精确性：每个采样点的深度图读数 = 该点真实深度+3mm
    (在5mm margin内)，应被判定为安全(0个危险点)。

    背景：开发过程中首次用低分辨率相机(fx=500)做此测试时，发现18/100个点被误判为
    危险，诊断后确认是100个3D点中有24组共享了同一像素(finger_thickness=8mm在z=1m
    处只对应约4像素的物理分辨率)，导致部分点读到的是"同像素中后写入的另一个点"的
    深度值而非自己的。这是真实传感器分辨率限制的真实反映，不是函数bug——用高分辨率
    相机消除像素重叠后，本测试验证了数值是精确的。
    """
    K_highres = np.array([[5000, 0, 3200], [0, 5000, 2400], [0, 0, 1]], dtype=np.float64)
    H, W = 4800, 6400

    grasp_cam = {"center": np.array([0., 0., 1.]), "v": np.array([0., 0., 1.]),
                 "u": np.array([1., 0., 0.]), "w": 0.06}
    gripper_cfg = {"finger_thickness": 0.008, "finger_length": 0.045, "approach_clearance": 0.06}
    collision_cfg = {"alpha": 3.0, "depth_occlusion_margin": 0.005}

    vols = CollisionVolumes(center=grasp_cam["center"], v_approach=grasp_cam["v"],
                              u_orientation=grasp_cam["u"], width=grasp_cam["w"],
                              finger_thickness=gripper_cfg["finger_thickness"],
                              finger_length=gripper_cfg["finger_length"],
                              approach_clearance=gripper_cfg["approach_clearance"])
    pts_fingers = vols.sample_points_fingers()
    pts_approach = vols.sample_points_approach()
    all_pts = np.concatenate([pts_fingers, pts_approach], axis=0)
    pixel_xy, expected_depth, valid = project_points_to_depth_pixels(all_pts, K_highres)

    unique_pixels, counts = np.unique(pixel_xy, axis=0, return_counts=True)
    assert len(unique_pixels) == len(all_pts), \
        f"高分辨率相机下仍有像素重叠({len(unique_pixels)}/{len(all_pts)})，测试前提不满足"

    depth_image = np.zeros((H, W))
    for i in range(len(all_pts)):
        col, row = pixel_xy[i]
        if 0 <= col < W and 0 <= row < H:
            depth_image[row, col] = expected_depth[i] + 0.003  # 在5mm margin内

    P = compute_collision_proxy_score(grasp_cam, depth_image, K_highres, gripper_cfg, collision_cfg)
    assert P > 0.99, f"像素不重叠条件下margin容差测试应得到P_coll≈1.0，实际={P}"
    print(f"[test_margin_tolerance_with_high_res_camera] P_coll={P:.4f}, PASS")


def test_margin_exceeded_with_high_res_camera():
    """对照组：超出margin(10mm更近的遮挡)的情形，同样用高分辨率相机消除离散化干扰"""
    K_highres = np.array([[5000, 0, 3200], [0, 5000, 2400], [0, 0, 1]], dtype=np.float64)
    H, W = 4800, 6400

    grasp_cam = {"center": np.array([0., 0., 1.]), "v": np.array([0., 0., 1.]),
                 "u": np.array([1., 0., 0.]), "w": 0.06}
    gripper_cfg = {"finger_thickness": 0.008, "finger_length": 0.045, "approach_clearance": 0.06}
    collision_cfg = {"alpha": 3.0, "depth_occlusion_margin": 0.005}

    vols = CollisionVolumes(center=grasp_cam["center"], v_approach=grasp_cam["v"],
                              u_orientation=grasp_cam["u"], width=grasp_cam["w"],
                              finger_thickness=gripper_cfg["finger_thickness"],
                              finger_length=gripper_cfg["finger_length"],
                              approach_clearance=gripper_cfg["approach_clearance"])
    pts_fingers = vols.sample_points_fingers()
    pts_approach = vols.sample_points_approach()
    all_pts = np.concatenate([pts_fingers, pts_approach], axis=0)
    pixel_xy, expected_depth, valid = project_points_to_depth_pixels(all_pts, K_highres)

    depth_image = np.zeros((H, W))
    for i in range(len(all_pts)):
        col, row = pixel_xy[i]
        if 0 <= col < W and 0 <= row < H:
            depth_image[row, col] = expected_depth[i] - 0.010  # 比每个点自身深度近10mm，超过5mm margin

    P = compute_collision_proxy_score(grasp_cam, depth_image, K_highres, gripper_cfg, collision_cfg)
    assert P < 0.1, f"超出margin的遮挡场景P_coll应显著低，实际={P}"
    print(f"[test_margin_exceeded_with_high_res_camera] P_coll={P:.4f}, PASS")


if __name__ == "__main__":
    test_pinhole_projection_basic()
    test_collision_score_no_occlusion()
    test_collision_score_full_occlusion()
    test_margin_tolerance_with_high_res_camera()
    test_margin_exceeded_with_high_res_camera()
    print("=" * 60)
    print("所有 collision_pruning 单元测试通过！")

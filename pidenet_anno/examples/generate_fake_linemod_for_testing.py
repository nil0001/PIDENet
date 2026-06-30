# -*- coding: utf-8 -*-
"""
examples/generate_fake_linemod_for_testing.py
=================================================
生成一套符合LINEMOD目录结构和yml格式的【模拟】数据（一个简化的"花生形"CAD模型 +
3帧随机位姿的depth/mask/rgb/gt.yml/info.yml），用于在没有真实LINEMOD数据集时，
快速自检本项目代码能否在你的环境中正常运行（依赖库是否齐全、路径解析是否正确等）。

【这不是真实LINEMOD数据，不能用它生成的标签训练任何实际模型】，
仅用于验证代码能跑通。生成的数据会写入 ./fake_linemod_test_data/ 目录下。

用法：
  cd 项目根目录
  python examples/generate_fake_linemod_for_testing.py
  然后修改 configs/config.yaml 中的 dataset_root 为
  "./fake_linemod_test_data/Linemod_preprocessed"，运行 python run_object01_demo.py
  确认整条流程能跑通后，再把 dataset_root 改回你的真实LINEMOD路径正式处理。
"""
import os
import sys
sys.path.insert(0, ".")
import numpy as np
import trimesh
import cv2
import yaml

ROOT = "./fake_linemod_test_data/Linemod_preprocessed"
os.makedirs(f"{ROOT}/data/01/depth", exist_ok=True)
os.makedirs(f"{ROOT}/data/01/mask", exist_ok=True)
os.makedirs(f"{ROOT}/data/01/rgb", exist_ok=True)
os.makedirs(f"{ROOT}/models", exist_ok=True)
os.makedirs(f"{ROOT}/segnet_results/01_label", exist_ok=True)

# 1. 生成花生形PLY模型(毫米单位,符合LINEMOD官方惯例)，保存为obj_01.ply
sphere1 = trimesh.creation.icosphere(radius=35, subdivisions=3)
sphere1.apply_translation([-40, 0, 0])
sphere2 = trimesh.creation.icosphere(radius=35, subdivisions=3)
sphere2.apply_translation([40, 0, 0])
neck = trimesh.creation.cylinder(radius=15, height=100, sections=48)
R_neck = trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0])[:3, :3]
neck.vertices = (R_neck @ neck.vertices.T).T
peanut_mm = sphere1.union(neck).union(sphere2)
peanut_mm.export(f"{ROOT}/models/obj_01.ply")
print("已生成 obj_01.ply, 顶点数:", len(peanut_mm.vertices), "bounds(mm):", peanut_mm.bounds)

# 2. 生成3帧的gt.yml (cam_R_m2c, cam_t_m2c均为毫米单位的官方约定)
np.random.seed(0)
gt_dict = {}
for i in range(3):
    # 随机生成一个合理的旋转矩阵(用随机轴角)
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.uniform(0, 2*np.pi)
    K_mat = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    R = np.eye(3) + np.sin(angle)*K_mat + (1-np.cos(angle))*(K_mat@K_mat)

    t_mm = np.array([np.random.uniform(-50,50), np.random.uniform(-50,50), np.random.uniform(400,600)])

    gt_dict[i] = [{
        "cam_R_m2c": R.flatten().tolist(),
        "cam_t_m2c": t_mm.tolist(),
        "obj_bb": [100, 100, 200, 150],
        "obj_id": 1,
    }]

with open(f"{ROOT}/data/01/gt.yml", "w") as f:
    yaml.dump(gt_dict, f, default_flow_style=None)
print("已生成 gt.yml, 3帧")

# 3. 生成 info.yml (相机内参,3帧用相同K)
K_cam = [572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0]
info_dict = {i: {"cam_K": K_cam, "depth_scale": 1.0} for i in range(3)}
with open(f"{ROOT}/data/01/info.yml", "w") as f:
    yaml.dump(info_dict, f, default_flow_style=None)
print("已生成 info.yml")

# 4. 生成3帧的深度图(简化:用一个固定背景深度+物体投影区域的深度做近似模拟)
H, W = 480, 640
K_mat3x3 = np.array(K_cam).reshape(3,3)
fx, fy, cx, cy = K_mat3x3[0,0], K_mat3x3[1,1], K_mat3x3[0,2], K_mat3x3[1,2]

for i in range(3):
    entry = gt_dict[i][0]
    R = np.array(entry["cam_R_m2c"]).reshape(3,3)
    t_mm = np.array(entry["cam_t_m2c"])
    t_m = t_mm / 1000.0

    # 用真实的物体顶点(转米)+变换到相机系+投影,生成一张近似深度图
    verts_m = peanut_mm.vertices / 1000.0
    verts_cam = (R @ verts_m.T).T + t_m

    depth_img = np.zeros((H, W), dtype=np.uint16)
    depth_img[:] = int(2.0 * 1000)  # 背景填充2米(毫米单位存储,符合LINEMOD惯例)

    valid_z = verts_cam[:, 2] > 0
    cols = np.round(verts_cam[valid_z, 0] * fx / verts_cam[valid_z, 2] + cx).astype(int)
    rows = np.round(verts_cam[valid_z, 1] * fy / verts_cam[valid_z, 2] + cy).astype(int)
    zs_mm = (verts_cam[valid_z, 2] * 1000).astype(np.uint16)

    in_bounds = (cols >= 0) & (cols < W) & (rows >= 0) & (rows < H)
    depth_img[rows[in_bounds], cols[in_bounds]] = zs_mm[in_bounds]

    cv2.imwrite(f"{ROOT}/data/01/depth/{i:04d}.png", depth_img)

    # mask: 简单地把"非背景深度"的位置标记为前景
    mask_img = np.where(depth_img < 1999, 255, 0).astype(np.uint8)
    cv2.imwrite(f"{ROOT}/data/01/mask/{i:04d}.png", mask_img)

    # rgb: 占位图(本测试不依赖rgb内容)
    rgb_img = np.full((H, W, 3), 128, dtype=np.uint8)
    cv2.imwrite(f"{ROOT}/data/01/rgb/{i:04d}.png", rgb_img)

print("已生成3帧 depth/mask/rgb")
print("模拟数据集生成完毕:", ROOT)

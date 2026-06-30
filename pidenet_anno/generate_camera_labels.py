# -*- coding: utf-8 -*-
"""
generate_camera_labels.py
============================
顶层脚本：给定物体坐标系下的候选抓取列表(来自annotation/pipeline.py)，
对该物体在LINEMOD数据集中出现的每一帧图像，生成相机坐标系下的抓取标签。

流程（对每一帧）：
  1. 从gt.yml读取该帧的 cam_R_m2c, cam_t_m2c
  2. 用 utils/transform.py 把物体系候选抓取(center, v, u, w)变换到相机系
  3. 从 info.yml 读取相机内参K，从depth.png读取该帧深度图
  4. 用 annotation/collision_pruning.py 计算该候选在该帧观测下的碰撞代理分数 P_coll
  5. 最终分数 S = Q(g_k) * P_coll  (论文Eq.11的离线几何代理版本)
  6. 每帧保留Top-N个候选(按S降序)，写入输出字典

最终用 utils/io_utils.py 的 GraspLabelDumper 写出用户指定格式的yml文件。
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.io_utils import (
    LinemodPaths, load_gt_yml, load_info_yml, load_depth_image, GraspLabelDumper
)
from utils.transform import batch_transform_grasps_obj_to_cam
from annotation.collision_pruning import compute_collision_proxy_score


def generate_all_frame_labels(object_candidates, paths: LinemodPaths, config: dict, verbose=True):
    """
    object_candidates: list[GraspCandidate]，来自annotation.pipeline.generate_object_frame_candidates
                        （物体坐标系下，已含Q(g_k)分数）
    paths: LinemodPaths对象，提供gt.yml/info.yml/depth路径
    config: 完整config字典

    返回: dict {frame_idx(int): [pose_dict, pose_dict, ...]}
          每个pose_dict含键: w, v(3,), u(3,), center(3,), S
          （这正是 GraspLabelDumper.dump 期望的输入格式）
    """
    gripper_cfg = config["gripper"]
    collision_cfg = config["collision"]
    frames_cfg = config["frames"]

    if verbose:
        print("=" * 70)
        print("[generate_all_frame_labels] 读取 gt.yml / info.yml")
    gt_data = load_gt_yml(paths.gt_yml, target_obj_id=config["object_id"], verbose=verbose)
    info_data = load_info_yml(paths.info_yml, verbose=verbose)

    frame_indices = sorted(gt_data.keys())
    if not frames_cfg["process_all"]:
        frame_indices = frame_indices[: frames_cfg["max_frames"]]
        if verbose:
            print(f"[generate_all_frame_labels] process_all=false, 仅处理前{len(frame_indices)}帧(调试模式)")

    # 把物体坐标系候选打包为transform.py期望的字典格式，预先转换一次，避免每帧重复转换类型
    candidates_obj_dicts = []
    for cand in object_candidates:
        if not cand.score.feasible:
            continue  # 不可行的候选(Q=0)不参与后续相机系标签生成
        candidates_obj_dicts.append({
            "center": cand.geom.center,
            "v": cand.geom.v,
            "u": cand.geom.u,
            "w": cand.geom.width,
            "Q": cand.score.Q,
        })

    if len(candidates_obj_dicts) == 0:
        raise RuntimeError(
            "object_candidates中没有任何feasible=True的候选，无法生成相机系标签。"
            "请检查annotation/pipeline.py的输出，或调整config.yaml中的摩擦系数/打分权重。"
        )

    if verbose:
        print(f"[generate_all_frame_labels] 物体系可行候选数={len(candidates_obj_dicts)}, "
              f"待处理帧数={len(frame_indices)}")

    frame_labels = {}
    skipped_frames = []

    for fi, frame_idx in enumerate(frame_indices):
        gt_entry = gt_data[frame_idx]
        R, t = gt_entry["R"], gt_entry["t"]

        if frame_idx in info_data and info_data[frame_idx] is not None:
            K = info_data[frame_idx]["K"]
            depth_scale = info_data[frame_idx]["depth_scale"]
        elif info_data.get("_K_common") is not None:
            K = info_data["_K_common"]
            depth_scale = 1.0
        else:
            skipped_frames.append(frame_idx)
            continue

        frame_paths = paths.frame_paths(frame_idx)
        if not os.path.exists(frame_paths["depth"]):
            skipped_frames.append(frame_idx)
            continue

        try:
            depth_image = load_depth_image(frame_paths["depth"], depth_scale=depth_scale)
        except Exception as e:
            if verbose:
                print(f"  ⚠ 帧{frame_idx}深度图读取失败: {e}，跳过")
            skipped_frames.append(frame_idx)
            continue

        cam_frame_candidates = batch_transform_grasps_obj_to_cam(candidates_obj_dicts, R, t)

        pose_dicts_this_frame = []
        for cand_cam in cam_frame_candidates:
            try:
                P_coll = compute_collision_proxy_score(
                    cand_cam, depth_image, K, gripper_cfg, collision_cfg, verbose=False
                )
            except Exception as e:
                P_coll = 0.5  # 计算异常时保守取中性值，不让单个候选的问题影响整帧输出
                if verbose and fi < 3:  # 只在前几帧打印详细错误，避免日志爆炸
                    print(f"    ⚠ 帧{frame_idx}碰撞分数计算异常: {e}，该候选P_coll取保守值0.5")

            S_final = cand_cam["Q"] * P_coll  # 论文Eq.11: Q~(g_k) = Q(g_k) * P_coll(g_k)

            pose_dicts_this_frame.append({
                "w": float(cand_cam["w"]),
                "v": cand_cam["v"].tolist() if hasattr(cand_cam["v"], "tolist") else list(cand_cam["v"]),
                "u": cand_cam["u"].tolist() if hasattr(cand_cam["u"], "tolist") else list(cand_cam["u"]),
                "center": cand_cam["center"].tolist() if hasattr(cand_cam["center"], "tolist") else list(cand_cam["center"]),
                "S": float(S_final),
            })

        pose_dicts_this_frame.sort(key=lambda p: p["S"], reverse=True)
        frame_labels[frame_idx] = pose_dicts_this_frame

        if verbose and (fi % max(1, len(frame_indices) // 10) == 0 or fi == len(frame_indices) - 1):
            print(f"  进度: {fi+1}/{len(frame_indices)} 帧已处理 (当前帧{frame_idx}, "
                  f"候选数={len(pose_dicts_this_frame)}, "
                  f"最高分={pose_dicts_this_frame[0]['S']:.4f})")

    if verbose:
        print(f"\n[generate_all_frame_labels] 完成。成功处理{len(frame_labels)}帧, "
              f"跳过{len(skipped_frames)}帧(深度图缺失/读取失败)")
        if skipped_frames:
            print(f"  跳过的帧号示例: {skipped_frames[:10]}")

    return frame_labels


def run_camera_label_generation(object_candidates, config: dict, verbose=True):
    """顶层封装：从config构造LinemodPaths，调用generate_all_frame_labels，写出yml文件"""
    paths = LinemodPaths(config["dataset_root"], config["object_id"])
    ok, missing = paths.check_exists(verbose=verbose)
    if not ok:
        raise FileNotFoundError(
            f"数据集路径检查未通过，缺失: {missing}。请检查config.yaml中的dataset_root设置。"
        )

    frame_labels = generate_all_frame_labels(object_candidates, paths, config, verbose=verbose)

    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"grasp_labels_{paths.obj_str}.yml")
    GraspLabelDumper.dump(frame_labels, out_path)

    return frame_labels, out_path

# -*- coding: utf-8 -*-
"""
run_object01_demo.py
=======================
一键运行脚本：复现论文PIDENet第3.2节的离线抓取标注流程，针对LINEMOD 01号物体(ape)。

运行步骤：
  1. 加载配置 configs/config.yaml
  2. 读取 obj_01.ply CAD模型
  3. [论文3.2.1~3.2.5] 生成物体坐标系下的候选抓取(annotation/pipeline.py)
  4. 保存物体坐标系候选为yml文件
  5. 3D可视化候选抓取(matplotlib静态图，附带open3d交互窗口/截图尝试)
  6. [相机系标签生成] 对该物体在LINEMOD中出现的所有帧，生成相机坐标系抓取标签
  7. 保存最终标签为 grasp_labels_01.yml

运行方式：
  python run_object01_demo.py
  (确保已正确设置 configs/config.yaml 中的 dataset_root)
"""

import sys
import os
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.io_utils import LinemodPaths, load_ply_model, load_config
from annotation.pipeline import generate_object_frame_candidates
from visualize_3d import visualize_candidates_matplotlib, visualize_candidates_open3d
from generate_camera_labels import run_camera_label_generation


def save_object_frame_candidates_yml(candidates, out_path):
    """
    将物体坐标系下的候选抓取保存为yml文件，方便单独检查/复用
    （不同于最终的相机系标签输出，这一步是中间产物，格式上额外保留了打分明细）
    """
    lines = []
    for i, cand in enumerate(candidates):
        geom = cand.geom
        score = cand.score
        lines.append(f"{i}:")
        lines.append(f"  source_pose_id: {cand.source_pose_id}")
        lines.append(f"  source_type: {cand.source_type}")
        lines.append(f"  center: [{geom.center[0]:.6f}, {geom.center[1]:.6f}, {geom.center[2]:.6f}]")
        lines.append(f"  v_approach: [{geom.v[0]:.6f}, {geom.v[1]:.6f}, {geom.v[2]:.6f}]")
        lines.append(f"  u_orientation: [{geom.u[0]:.6f}, {geom.u[1]:.6f}, {geom.u[2]:.6f}]")
        lines.append(f"  width: {geom.width:.6f}")
        lines.append(f"  Q: {score.Q:.6f}")
        lines.append(f"  feasible: {score.feasible}")
        lines.append(f"  S_geo: {score.S_geo:.6f}")
        lines.append(f"  S_ali: {score.S_ali:.6f}")
        lines.append(f"  S_wid: {score.S_wid:.6f}")
        lines.append(f"  S_com: {score.S_com:.6f}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[save_object_frame_candidates_yml] 已保存: {out_path}")


def _open3d_offscreen_likely_supported():
    """
    粗略检测当前环境是否可能支持open3d的离屏/窗口渲染。

    【为何需要这个检测，而不是简单依赖try-except】
    开发过程中发现：在缺少显示环境支持的容器中直接调用open3d的Visualizer.create_window
    (即便visible=False)，底层GLFW/OSMesa初始化失败后会触发C++层的Segmentation fault，
    而不是Python异常——这意味着try-except完全无法捕获该错误，会导致整个脚本进程直接
    崩溃退出，后续步骤(包括已经生成好的物体系候选、即将生成的相机系标签)全部不会执行。
    因此必须在调用前主动检测，宁可保守跳过，也不能让脚本崩溃丢失已完成的工作。

    检测逻辑：检查 XDG_RUNTIME_DIR 环境变量（多数Linux桌面/Xvfb环境会设置），
    以及 DISPLAY 环境变量（X11环境标志）。两者都缺失时，认为当前很可能是纯headless
    容器（无Xvfb包装），跳过open3d渲染，仅依赖已验证稳定的matplotlib静态图。
    """
    import os as _os
    has_xdg = bool(_os.environ.get("XDG_RUNTIME_DIR", "").strip())
    has_display = bool(_os.environ.get("DISPLAY", "").strip())
    return has_xdg or has_display


def main():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "config.yaml")
    config = load_config(config_path)

    print("=" * 70)
    print("PIDENet 抓取标注复现 — LINEMOD 物体处理")
    print("=" * 70)
    print(f"数据集路径: {config['dataset_root']}")
    print(f"目标物体: {config['object_id']:02d}")
    print()

    paths = LinemodPaths(config["dataset_root"], config["object_id"])
    print("[Step 0] 检查数据集路径...")
    ok, missing = paths.check_exists(verbose=True)
    if not ok:
        print("❌ 路径检查未通过，请先修改 configs/config.yaml 中的 dataset_root，然后重新运行。")
        print(f"   缺失项: {missing}")
        sys.exit(1)

    os.makedirs(config["output_dir"], exist_ok=True)

    print("\n[Step 1] 读取CAD模型...")
    mesh = load_ply_model(paths.ply_path, verbose=True)

    print("\n[Step 2] 生成物体坐标系候选抓取 (论文3.2.1~3.2.5节)...")
    candidates = generate_object_frame_candidates(mesh, config, verbose=True)

    if len(candidates) == 0:
        print("❌ 未能生成任何候选抓取，请检查CAD模型是否正常、config.yaml参数是否合理。")
        sys.exit(1)

    feasible_count = sum(1 for c in candidates if c.score.feasible)
    print(f"\n候选生成完毕: 总数={len(candidates)}, 可行数={feasible_count}")
    if feasible_count == 0:
        print("⚠ 警告: 没有任何可行候选(所有Q=0)。可能原因: 摩擦系数过小/夹爪宽度范围设置不合理/"
              "物体几何特征不适合简单凸性缺陷分析。标签生成将无法进行，请调整config.yaml后重试。")
        sys.exit(1)

    obj_candidates_path = os.path.join(
        config["output_dir"], f"object_{paths.obj_str}_grasp_candidates_objframe.yml"
    )
    save_object_frame_candidates_yml(candidates, obj_candidates_path)

    print("\n[Step 3] 3D可视化候选抓取...")
    matplotlib_path = os.path.join(
        config["output_dir"], f"object_{paths.obj_str}_grasp_candidates_3d.png"
    )
    visualize_candidates_matplotlib(mesh, candidates, top_n=10, arrow_length=0.15 * (
        np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    ), save_path=matplotlib_path)

    open3d_screenshot_path = os.path.join(
        config["output_dir"], f"object_{paths.obj_str}_grasp_candidates_open3d.png"
    )
    if _open3d_offscreen_likely_supported():
        try:
            visualize_candidates_open3d(
                mesh, candidates, top_n=10,
                arrow_length=0.15 * (np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])),
                save_path=open3d_screenshot_path, show_window=False
            )
        except Exception as e:
            print(f"  ⚠ open3d离屏渲染失败(详见README\"已知环境兼容性问题\"一节): {e}")
            print(f"  matplotlib静态图已生成于: {matplotlib_path}，可作为可靠的可视化结果。")
    else:
        print(f"  ⚠ 检测到当前环境缺少 XDG_RUNTIME_DIR/DISPLAY 环境变量(可能是纯headless容器，"
              f"如未经Xvfb包装的Docker/CI环境)，跳过open3d离屏渲染以避免底层段错误风险。")
        print(f"  matplotlib静态图已生成于: {matplotlib_path}，可作为可靠的可视化结果。")
        print(f"  若你在普通Windows/Mac桌面环境运行本脚本，通常会有正常的GUI支持，"
              f"open3d渲染应能正常工作；如仍需在当前环境强行尝试，可参考"
              f"用 `xvfb-run -a python run_object01_demo.py` 包装运行命令。")

    print(f"\n  如需交互式3D窗口查看(本机有显示器时推荐)，可单独运行:")
    print(f'  python -c "from visualize_3d import *; from utils.io_utils import *; '
          f"mesh=load_ply_model('{paths.ply_path}'); "
          f"from annotation.pipeline import generate_object_frame_candidates; "
          f"import yaml; config=yaml.safe_load(open('configs/config.yaml')); "
          f"candidates=generate_object_frame_candidates(mesh, config); "
          f'visualize_candidates_open3d(mesh, candidates, show_window=True)"')

    print("\n[Step 4] 生成所有帧的相机坐标系抓取标签...")
    frame_labels, labels_out_path = run_camera_label_generation(candidates, config, verbose=True)

    print("\n" + "=" * 70)
    print("全部完成！输出文件汇总:")
    print(f"  - 物体系候选(含打分明细): {obj_candidates_path}")
    print(f"  - 3D可视化(matplotlib，保证可用): {matplotlib_path}")
    print(f"  - 3D可视化(open3d截图，环境受限时可能失败): {open3d_screenshot_path}")
    print(f"  - 相机系全帧抓取标签: {labels_out_path}")
    print(f"  - 处理帧数: {len(frame_labels)}")
    print("=" * 70)


if __name__ == "__main__":
    main()

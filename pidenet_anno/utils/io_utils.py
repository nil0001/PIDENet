# -*- coding: utf-8 -*-
"""
utils/io_utils.py
==================
负责所有磁盘IO：读取CAD模型(PLY)、gt.yml位姿标注、info.yml相机内参、
depth/mask/rgb图像，以及对LINEMOD数据集格式的自检。

【重要】因为开发环境无法访问真实LINEMOD数据集，本模块在每个读取函数中
都加入了格式自检与详细print诊断信息。第一次运行时请仔细查看终端输出，
确认以下假设是否与你的真实数据吻合：
  1. PLY 顶点单位是毫米还是米（LINEMOD官方通常是毫米）
  2. mask.png 是二值图(0/255)还是多类语义图(像素值=类别ID)
  3. depth.png 是16位单通道，单位毫米 * depth_scale
  4. gt.yml 中每帧可能对应多个物体实例（列表），需要按 obj_id 筛选
"""

import os
import yaml
import numpy as np
import cv2

try:
    import trimesh
except ImportError:
    trimesh = None


# ==============================================================================
# 路径管理
# ==============================================================================
class LinemodPaths:
    """根据 dataset_root 和 object_id 统一生成各类文件路径，并做存在性检查。"""

    def __init__(self, dataset_root: str, object_id: int):
        self.root = dataset_root
        self.obj_id = object_id
        self.obj_str = f"{object_id:02d}"

        self.data_dir = os.path.join(dataset_root, "data", self.obj_str)
        self.depth_dir = os.path.join(self.data_dir, "depth")
        self.mask_dir = os.path.join(self.data_dir, "mask")
        self.rgb_dir = os.path.join(self.data_dir, "rgb")
        self.gt_yml = os.path.join(self.data_dir, "gt.yml")
        self.info_yml = os.path.join(self.data_dir, "info.yml")

        self.models_dir = os.path.join(dataset_root, "models")
        self.ply_path = os.path.join(self.models_dir, f"obj_{self.obj_str}.ply")

        self.segnet_dir = os.path.join(
            dataset_root, "segnet_results", f"{self.obj_str}_label"
        )

    def check_exists(self, verbose=True):
        """检查关键路径是否存在，返回 (ok: bool, missing: list[str])"""
        targets = {
            "data目录": self.data_dir,
            "depth目录": self.depth_dir,
            "mask目录": self.mask_dir,
            "rgb目录": self.rgb_dir,
            "gt.yml": self.gt_yml,
            "info.yml": self.info_yml,
            "models目录": self.models_dir,
            f"obj_{self.obj_str}.ply": self.ply_path,
        }
        missing = []
        for name, path in targets.items():
            exists = os.path.exists(path)
            if verbose:
                flag = "✓" if exists else "✗ 缺失"
                print(f"  [{flag}] {name}: {path}")
            if not exists:
                missing.append(name)
        ok = len(missing) == 0
        if verbose:
            if ok:
                print("  => 路径检查通过，所有必需文件/目录均存在。\n")
            else:
                print(f"  => 警告：以下路径缺失，请检查 config.yaml 中的 dataset_root 设置: {missing}\n")
        return ok, missing

    def frame_paths(self, frame_idx: int):
        """给定帧号，返回该帧 depth/mask/rgb 的文件路径（统一4位数字命名）"""
        name = f"{frame_idx:04d}.png"
        return {
            "depth": os.path.join(self.depth_dir, name),
            "mask": os.path.join(self.mask_dir, name),
            "rgb": os.path.join(self.rgb_dir, name),
        }

    def num_frames_available(self):
        """统计 depth 目录下实际有多少帧（用于校验与gt.yml条目数是否一致）"""
        if not os.path.isdir(self.depth_dir):
            return 0
        files = [f for f in os.listdir(self.depth_dir) if f.lower().endswith(".png")]
        return len(files)


# ==============================================================================
# CAD 模型读取
# ==============================================================================
def load_ply_model(ply_path: str, assume_unit="auto", verbose=True):
    """
    读取 PLY 模型，返回 trimesh.Trimesh 对象，顶点坐标统一转换为【米】。

    LINEMOD 官方模型 (obj_XX.ply) 顶点坐标通常以毫米为单位。本函数通过
    检测模型对角线长度做启发式判断：
      - 若对角线 > 10  （数值上），认为单位是毫米，自动 / 1000 转米
      - 若对角线 <= 10，认为已经是米，不做转换
    LINEMOD物体实际物理尺寸大多在 3cm~30cm 之间，因此：
      - 毫米单位下对角线数值约为 30~300
      - 米单位下对角线数值约为 0.03~0.30
    这两个区间在数值上差3个数量级，用阈值10做区分是安全的。

    assume_unit: "auto" | "mm" | "m"  — 若你已知确切单位，建议显式指定以跳过启发式判断
    """
    if trimesh is None:
        raise ImportError("需要 trimesh 库：pip install trimesh")

    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"找不到PLY模型文件: {ply_path}")

    mesh = trimesh.load(ply_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        # 某些PLY导出工具会包裹成Scene，取第一个几何体
        geoms = list(mesh.geometry.values())
        if len(geoms) == 0:
            raise ValueError(f"PLY文件 {ply_path} 不包含任何几何体")
        mesh = geoms[0]

    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))

    if assume_unit == "mm":
        scale = 1.0 / 1000.0
        detected = "mm（手动指定）"
    elif assume_unit == "m":
        scale = 1.0
        detected = "m（手动指定）"
    else:
        if diag > 10.0:
            scale = 1.0 / 1000.0
            detected = f"mm（自动检测，对角线原始值={diag:.3f}）"
        else:
            scale = 1.0
            detected = f"m（自动检测，对角线原始值={diag:.3f}）"

    mesh.apply_scale(scale)
    new_diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))

    if verbose:
        print(f"[load_ply_model] 文件: {ply_path}")
        print(f"  顶点数={len(mesh.vertices)}, 面数={len(mesh.faces)}")
        print(f"  检测单位: {detected}")
        print(f"  缩放后对角线长度: {new_diag*100:.2f} cm")
        print(f"  缩放后 bounding box (米): min={mesh.bounds[0]}, max={mesh.bounds[1]}")
        if not (0.01 < new_diag < 1.0):
            print(f"  ⚠ 警告：物体对角线{new_diag*100:.1f}cm超出常见LINEMOD物体尺寸范围(3~30cm)，"
                  f"请用 assume_unit 参数手动确认单位是否判断正确！")

    return mesh


# ==============================================================================
# YAML 标注读取（gt.yml / info.yml）
# ==============================================================================
def load_gt_yml(gt_yml_path: str, target_obj_id: int, verbose=True):
    """
    读取 gt.yml，返回 dict: {frame_idx(int): {"R": 3x3 ndarray, "t": 3, ndarray(米), "bb": [...]}}

    LINEMOD的 gt.yml 中每一帧的值是一个【列表】（因为同一帧可能出现多个物体实例，
    尤其是在 occlusion LINEMOD 变体中）。本函数会在该帧列表中查找 obj_id == target_obj_id
    的那一项；若同一帧出现多次匹配（理论上不应该，但做容错），取第一个并打印警告。

    cam_t_m2c 在官方数据集中以【毫米】为单位，本函数会自动转换为【米】。
    """
    if not os.path.exists(gt_yml_path):
        raise FileNotFoundError(f"找不到 gt.yml: {gt_yml_path}")

    with open(gt_yml_path, "r") as f:
        raw = yaml.safe_load(f)

    result = {}
    multi_match_frames = []
    no_match_frames = []

    for frame_key, entries in raw.items():
        frame_idx = int(frame_key)
        # 兼容两种可能结构：list[dict] 或单个dict（极少数变体数据集可能简化成单实例）
        if isinstance(entries, dict):
            entries = [entries]

        matched = [e for e in entries if int(e.get("obj_id", -1)) == target_obj_id]

        if len(matched) == 0:
            no_match_frames.append(frame_idx)
            continue
        if len(matched) > 1:
            multi_match_frames.append(frame_idx)

        e = matched[0]
        R = np.array(e["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
        t_mm = np.array(e["cam_t_m2c"], dtype=np.float64).reshape(3)
        t_m = t_mm / 1000.0  # 毫米 -> 米
        bb = e.get("obj_bb", None)

        result[frame_idx] = {"R": R, "t": t_m, "bb": bb}

    if verbose:
        print(f"[load_gt_yml] 文件: {gt_yml_path}")
        print(f"  总帧数(yml条目)={len(raw)}, 成功匹配obj_id={target_obj_id}的帧数={len(result)}")
        if no_match_frames:
            print(f"  ⚠ 有 {len(no_match_frames)} 帧未找到该物体实例(可能是occlusion数据集变体下"
                  f"该物体在该帧不可见)，示例: {no_match_frames[:5]}")
        if multi_match_frames:
            print(f"  ⚠ 有 {len(multi_match_frames)} 帧匹配到多个同 obj_id 实例，"
                  f"已默认取第一个，示例: {multi_match_frames[:5]}")
        if len(result) > 0:
            sample_frame = sorted(result.keys())[0]
            print(f"  样例帧 {sample_frame}: t(米)={result[sample_frame]['t']}, "
                  f"R前两行={result[sample_frame]['R'][:2]}")

    return result


def load_info_yml(info_yml_path: str, verbose=True):
    """
    读取 info.yml，返回 dict: {frame_idx(int): {"K": 3x3 ndarray, "depth_scale": float}}
    若所有帧相机内参相同（LINEMOD常见情况），额外返回一个 "K_common" 字段方便直接取用。
    """
    if not os.path.exists(info_yml_path):
        raise FileNotFoundError(f"找不到 info.yml: {info_yml_path}")

    with open(info_yml_path, "r") as f:
        raw = yaml.safe_load(f)

    result = {}
    for frame_key, e in raw.items():
        frame_idx = int(frame_key)
        K = np.array(e["cam_K"], dtype=np.float64).reshape(3, 3)
        depth_scale = float(e.get("depth_scale", 1.0))
        result[frame_idx] = {"K": K, "depth_scale": depth_scale}

    # 检查是否所有帧内参一致
    Ks = [v["K"] for v in result.values()]
    all_same = all(np.allclose(Ks[0], k) for k in Ks) if len(Ks) > 0 else False

    if verbose:
        print(f"[load_info_yml] 文件: {info_yml_path}, 总帧数={len(result)}")
        if len(result) > 0:
            print(f"  样例K矩阵(第0帧):\n{Ks[0]}")
            print(f"  所有帧内参是否一致: {all_same}")

    result["_K_common"] = Ks[0] if all_same and len(Ks) > 0 else None
    result["_all_same_K"] = all_same
    return result


# ==============================================================================
# 图像读取
# ==============================================================================
def load_depth_image(path: str, depth_scale: float = 1.0):
    """
    读取深度图，返回单位为【米】的 float64 ndarray (H, W)。

    LINEMOD depth.png 是16位无符号单通道PNG，像素值通常以【毫米】为单位，
    实际物理深度 = pixel_value * depth_scale / 1000.0 （米）
    depth_scale 一般从 info.yml 读取（常见值为1.0）。
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取深度图: {path}")
    if img.ndim == 3:
        img = img[:, :, 0]
    depth_m = img.astype(np.float64) * depth_scale / 1000.0
    return depth_m


def load_mask_image(path: str, expect_binary_value=255, verbose=False):
    """
    读取mask图，返回二值bool数组 (H, W)，True=该像素属于目标物体。

    LINEMOD的mask可能是以下两种格式之一：
      (a) 纯二值图：背景=0，物体=255（最常见）
      (b) 语义/实例标签图：背景=0，不同物体=不同灰度值(1,2,3...)

    本函数读取后会打印唯一像素值集合，便于你确认到底是哪种格式。
    若检测到多于2个唯一值，默认采用"非零即前景"的策略（适用于该mask文件本身
    就是针对单一物体生成的情况，这正是LINEMOD官方per-object mask目录的设计）。
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取mask图: {path}")
    if img.ndim == 3:
        img = img[:, :, 0]

    unique_vals = np.unique(img)
    if verbose:
        print(f"[load_mask_image] {path} 唯一像素值: {unique_vals[:10]}"
              f"{'...' if len(unique_vals) > 10 else ''}")

    binary = img > 0
    return binary


def load_rgb_image(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取RGB图: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ==============================================================================
# 配置文件读取
# ==============================================================================
def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ==============================================================================
# 输出YAML写入（自定义格式：pose1/pose2/... 而非yaml默认的列表格式）
# ==============================================================================
class GraspLabelDumper:
    """
    按用户要求的格式写出抓取标签 yml：
        0:
          pose1: {w: ..., v: [...], u: [...], center: [...], S: ...}
          pose2: {...}
        1:
          pose1: {...}
        ...
    PyYAML默认的dump会把嵌套dict展开成多行块状格式，不符合用户想要的flow-style单行格式，
    因此这里手写格式化逻辑而不是直接调用 yaml.dump()。
    """

    @staticmethod
    def _fmt_vec(v, ndigits=6):
        return "[" + ", ".join(f"{x:.{ndigits}f}" for x in v) + "]"

    @staticmethod
    def _fmt_pose_line(pose_dict, ndigits=6):
        w = pose_dict["w"]
        v = pose_dict["v"]
        u = pose_dict["u"]
        c = pose_dict["center"]
        S = pose_dict["S"]
        return (
            f"{{w: {w:.{ndigits}f}, "
            f"v: {GraspLabelDumper._fmt_vec(v, ndigits)}, "
            f"u: {GraspLabelDumper._fmt_vec(u, ndigits)}, "
            f"center: {GraspLabelDumper._fmt_vec(c, ndigits)}, "
            f"S: {S:.{ndigits}f}}}"
        )

    @staticmethod
    def dump(frame_to_poses: dict, out_path: str):
        """
        frame_to_poses: {frame_idx(int): [pose_dict, pose_dict, ...]}
        每个 pose_dict 含键: w(float), v(3,), u(3,), center(3,), S(float)
        """
        lines = []
        for frame_idx in sorted(frame_to_poses.keys()):
            poses = frame_to_poses[frame_idx]
            lines.append(f"{frame_idx}:")
            for i, pose in enumerate(poses, start=1):
                line = GraspLabelDumper._fmt_pose_line(pose)
                lines.append(f"  pose{i}: {line}")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[GraspLabelDumper] 标签已写出: {out_path} (共{len(frame_to_poses)}帧)")

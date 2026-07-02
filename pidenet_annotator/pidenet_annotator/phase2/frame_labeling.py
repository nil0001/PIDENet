"""Phase 2 core: transform object-frame Phase-1 candidates into each
frame's camera coordinate system, then compute a soft collision-aware
score S = Q_offline * P_coll(frame).

P_coll combines two frame-specific occlusion signals:

  (a) Self-occlusion via mesh raycasting.
      Place the CAD mesh into the camera frame using gt.yml's R,t. For
      each contact point p1, p2 (also transformed into the camera frame),
      shoot a ray from the camera origin towards the point and check
      whether the first mesh hit is at (approximately) that point's own
      depth, or comes in noticeably shallower — the latter means some
      other part of the same object is occluding this grasp region.

  (b) Scene-occlusion via the segmentation mask.
      Project pkm into image pixels using K. If the pixel falls outside
      the segmentation mask (foreground of the target object as seen in
      this frame), the grasp region is either scene-occluded by another
      object, or out-of-frame — either way it is not manipulable in this
      view.

Each signal produces a per-contact-point risk in [0,1]; the candidate's
overall risk is the mean over its two contact points and pkm. Following
Eq. (12) in the paper we then use an exponential decay
    P_coll = exp( -alpha * risk )
so the score decays continuously with the severity of the occlusion,
rather than being hard-thresholded. Final frame-specific score is:
    S = Q_offline * P_coll         (matches paper's Eq. 13).
"""
from __future__ import annotations
from pathlib import Path
from typing import Any

import numpy as np
import cv2
import trimesh
import yaml

from .dataset_io import (load_gt, load_info, load_phase1_candidates,
                          iter_frame_ids, frame_paths)


# ---------- geometry helpers ----------

def transform_point(p_obj: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """p_cam = R @ p_obj + t   for a single point or (N,3) batch."""
    p_obj = np.asarray(p_obj)
    if p_obj.ndim == 1:
        return R @ p_obj + t
    return p_obj @ R.T + t


def transform_direction(v_obj: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Unit direction (u, v, x) transforms by R only (translation drops)."""
    v_obj = np.asarray(v_obj)
    if v_obj.ndim == 1:
        return R @ v_obj
    return v_obj @ R.T


def project_point(p_cam: np.ndarray, K: np.ndarray) -> tuple[float, float]:
    """Pinhole projection to (col, row) pixel. Assumes p_cam[2] > 0."""
    x, y, z = p_cam
    if z <= 0:
        return float('nan'), float('nan')
    u = (K[0, 0] * x + K[0, 2] * z) / z
    v = (K[1, 1] * y + K[1, 2] * z) / z
    return float(u), float(v)


# ---------- occlusion signals ----------

def compute_self_occlusion_for_grasp(mesh_cam: trimesh.Trimesh,
                                       points_cam: np.ndarray,
                                       tol_mm: float = 3.0,
                                       occ_scale_factor: float = 10.0) -> np.ndarray:
    """Grasp-aware self-occlusion. `points_cam` is expected to be the three
    query points of ONE candidate: (pkm, p1, p2), in that order.

    Semantics: what actually blocks a grasp is a mesh surface that lies
    clearly IN FRONT of the entire grasped feature (from the camera's
    perspective). p1 and p2 sit on the two contact walls of the feature
    (for a hole grasp, opposite sides of a handle wall/loop; for an outer
    grasp, opposite sides of a concavity). Together they define a depth
    interval `[d_near, d_far]` in the camera frame that is INTRINSIC to
    the feature -- any mesh hits inside that interval, or within
    `tol_mm` of either end, are the feature itself and are NOT occluders.

    A per-point risk in [0,1] is returned for each of the three query
    points:
        risk = 0 if the point is visible (no mesh strictly in front of
               d_near beyond the tolerance, along that point's ray).
        risk grows with how much closer the extra front-hit is than
               d_near, saturating at 1 when the front-hit is
               `occ_scale_factor * tol_mm` closer.
    """
    points_cam = np.atleast_2d(points_cam).astype(float)
    n = points_cam.shape[0]
    dists = np.linalg.norm(points_cam, axis=1)
    if n == 0:
        return np.zeros(0)

    d_near = float(dists.min())
    d_far = float(dists.max())

    valid = dists > 1e-6
    dirs = np.zeros_like(points_cam)
    dirs[valid] = points_cam[valid] / dists[valid, None]
    origins = np.zeros_like(points_cam)

    try:
        locs, idx_ray, idx_tri = mesh_cam.ray.intersects_location(
            origins, dirs, multiple_hits=True)
    except Exception:
        return np.ones(n)

    hits_per_ray = [[] for _ in range(n)]
    for k, ri in enumerate(idx_ray):
        hits_per_ray[ri].append(np.linalg.norm(locs[k]))

    occ_scale = occ_scale_factor * tol_mm
    risk = np.zeros(n)
    for i in range(n):
        hits = sorted(hits_per_ray[i])
        if not hits:
            # ray missed the mesh entirely (rare: numerical grazing)
            risk[i] = 1.0
            continue
        # any front-hit strictly shallower than the whole feature (with
        # tolerance) counts as an occluder for the feature
        front_hits = [h for h in hits if h < d_near - tol_mm]
        if not front_hits:
            risk[i] = 0.0
            continue
        gap = d_near - front_hits[0]     # positive, how much occluder is in front
        risk[i] = float(np.clip(gap / occ_scale, 0.0, 1.0))
    return risk


def compute_self_occlusion(mesh_cam: trimesh.Trimesh,
                            points_cam: np.ndarray,
                            tol_mm: float = 3.0) -> np.ndarray:
    """Backwards-compatible wrapper that treats every query as an
    independent point (no grasp-feature grouping). Prefer the grasp-aware
    variant `compute_self_occlusion_for_grasp` when you have (pkm,p1,p2)
    of a single candidate."""
    return compute_self_occlusion_for_grasp(mesh_cam, points_cam, tol_mm)


def compute_scene_occlusion(mask_binary: np.ndarray, K: np.ndarray,
                              points_cam: np.ndarray,
                              margin_px: int = 2) -> np.ndarray:
    """Project each camera-frame point into pixel coords; return risk in
    [0,1] per point: 0 if the pixel is well inside the segmentation mask,
    1 if it's outside (scene-occluded / clipped)."""
    H, W = mask_binary.shape
    dilated = cv2.dilate(mask_binary.astype(np.uint8),
                          np.ones((2 * margin_px + 1, 2 * margin_px + 1),
                                   dtype=np.uint8))
    risk = np.ones(len(points_cam))
    for i, p in enumerate(points_cam):
        u, v = project_point(p, K)
        if np.isnan(u):
            continue
        c = int(round(u)); r = int(round(v))
        if 0 <= r < H and 0 <= c < W and dilated[r, c] > 0:
            risk[i] = 0.0
    return risk


def score_candidate_in_frame(cand: dict, R: np.ndarray, t: np.ndarray,
                              K: np.ndarray, mesh_cam: trimesh.Trimesh,
                              mask_binary: np.ndarray, hp: dict) -> dict:
    """Transform one Phase-1 candidate into this frame's camera coords
    and compute its collision-aware final score."""
    p1_obj = np.array(cand['contact_p1'])
    p2_obj = np.array(cand['contact_p2'])
    pkm_obj = np.array(cand['center_pkm'])
    u_obj = np.array(cand['orientation_vector_u'])
    v_obj = np.array(cand['approach_vector_v'])
    x_obj = np.array(cand['local_x_axis'])

    p1_cam = transform_point(p1_obj, R, t)
    p2_cam = transform_point(p2_obj, R, t)
    pkm_cam = transform_point(pkm_obj, R, t)
    u_cam = transform_direction(u_obj, R)
    v_cam = transform_direction(v_obj, R)
    x_cam = transform_direction(x_obj, R)

    query = np.stack([pkm_cam, p1_cam, p2_cam])
    # Self-occlusion is checked ONLY on pkm (the grasp center). pkm sits
    # inside the grasp gap (for a hole grasp, in the middle of the hole;
    # for an outer grasp, on the free-space side of the concavity), so it
    # is the geometrically correct proxy for "can the gripper approach
    # from the camera direction without something in the way". Checking
    # p1 or p2 directly would spuriously flag every hole grasp as
    # occluded, since the ray from the camera to a back-side contact
    # point must first hit the near-side wall of the very loop being
    # grasped.
    self_risk_pkm = compute_self_occlusion_for_grasp(
        mesh_cam, pkm_cam[None, :], tol_mm=hp['phase2']['self_occ_tol_mm'])[0]
    # Scene occlusion via mask is still checked at all three (pkm,p1,p2)
    # so that even if pkm projects into the mask, an out-of-mask contact
    # point still degrades the score.
    scene_risk = compute_scene_occlusion(
        mask_binary, K, query, margin_px=hp['phase2']['scene_occ_margin_px'])
    # Combine: pkm's self-occlusion joins pkm's scene-occlusion; p1/p2
    # only contribute their scene-occlusion signal.
    per_point_risk = np.array([max(self_risk_pkm, scene_risk[0]),
                                scene_risk[1], scene_risk[2]])
    frame_risk = float(per_point_risk.mean())
    alpha = float(hp['phase2']['alpha'])
    P_coll = float(np.exp(-alpha * frame_risk))
    Q_off = float(cand['Q_score'])
    S = Q_off * P_coll

    return dict(
        id=int(cand['id']),
        kind=cand['kind'],
        w=float(cand['width_mm']),
        u=u_cam.tolist(),
        v=v_cam.tolist(),
        center=pkm_cam.tolist(),
        contact_p1=p1_cam.tolist(),
        contact_p2=p2_cam.tolist(),
        local_x=x_cam.tolist(),
        S=S,
        Q_offline=Q_off,
        P_coll=P_coll,
        frame_risk=frame_risk,
        per_point_risk=per_point_risk.tolist(),
    )


# ---------- per-frame + per-object drivers ----------

def label_one_frame(candidates: list[dict], R: np.ndarray, t: np.ndarray,
                     K: np.ndarray, mesh: trimesh.Trimesh,
                     mask_path: Path, hp: dict) -> dict:
    """All candidates transformed and scored in one frame."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"mask not found: {mask_path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask_bin = (mask > 128).astype(np.uint8)

    # bake the mesh into the camera frame once for this frame
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    mesh_cam = mesh.copy()
    mesh_cam.apply_transform(T)

    frame_out = {}
    for i, c in enumerate(candidates):
        scored = score_candidate_in_frame(c, R, t, K, mesh_cam, mask_bin, hp)
        frame_out[f'pose{i+1}'] = scored
    return frame_out


def label_object_frames(dataset_root: str | Path, obj_id: int,
                         phase1_yml: str | Path, hp: dict,
                         frame_ids: list[int] | None = None,
                         progress_every: int = 100) -> dict:
    """Run Phase 2 over every requested frame of one object. Returns a
    {frame_id: {poseK: {w,u,v,center,S,...}}} dict, ready for YAML dump.
    """
    dataset_root = Path(dataset_root)
    ply_path = dataset_root / 'models' / f'obj_{obj_id:02d}.ply'
    mesh = trimesh.load(str(ply_path), process=False)
    gt = load_gt(dataset_root / 'data' / f'{obj_id:02d}' / 'gt.yml')
    info = load_info(dataset_root / 'data' / f'{obj_id:02d}' / 'info.yml')
    ph1 = load_phase1_candidates(phase1_yml)
    cands = ph1['candidates']

    if frame_ids is None:
        frame_ids = iter_frame_ids(gt)

    all_out = {}
    for k, fid in enumerate(frame_ids):
        if fid not in gt:
            continue
        g = gt[fid]
        if g['obj_id'] != obj_id:
            continue
        K = info[fid]['K']
        mask_path = dataset_root / 'data' / f'{obj_id:02d}' / 'mask' / f'{fid:04d}.png'
        frame_out = label_one_frame(cands, g['R'], g['t'], K, mesh, mask_path, hp)
        all_out[fid] = frame_out
        if progress_every and (k + 1) % progress_every == 0:
            print(f'    processed {k+1}/{len(frame_ids)} frames', flush=True)
    return all_out


def dump_phase2_yaml(labels: dict, out_path: Path):
    """Write labels dict as YAML with per-frame keys and per-pose subkeys."""
    with open(out_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(labels, f, sort_keys=False, default_flow_style=None)

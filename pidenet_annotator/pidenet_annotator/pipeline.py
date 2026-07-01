"""Phase-1 offline annotation pipeline: CAD model -> ranked grasp candidates
in the object's own canonical coordinate frame. Implements 3.2.1-3.2.5
(3.2.6 Gaussian propagation intentionally excluded per instruction).
"""
import numpy as np
import cv2
import yaml

from .stable_pose import compute_stable_poses
from .projection import raycast_topdown
from .grasp2d import fit_efd_contour, outer_branch_pairs, hole_branch_pair
from .approach_vector import SurfaceQuery, unrotate, build_grasp_frame
from .scoring import score_candidate


def load_hp(path='hyperparams.yml'):
    """Load hyperparameters from a YAML file.

    Path resolution (in order): (1) the exact path given, (2) relative to
    the current working directory, (3) relative to the project root
    (parent of this package). This makes the same call work from Windows
    Explorer, VS Code, or a terminal launched from any directory.
    """
    import os
    from pathlib import Path
    candidates = [Path(path)]
    if not Path(path).is_absolute():
        candidates.append(Path.cwd() / path)
        pkg_root = Path(__file__).resolve().parent.parent
        candidates.append(pkg_root / path)
    for p in candidates:
        if p.exists():
            with open(p, encoding='utf-8') as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(
        f"Could not find hyperparams file. Tried: "
        + ", ".join(str(p) for p in candidates))


def _lookup_xyz(info, xy_pixel):
    """xy_pixel = (col, row) pixel-index coordinates, exactly as produced by
    cv2 contours / EFD reconstruction on the projection grid (OpenCV's
    point convention is (x=col, y=row)). Returns the full 3D world-space
    point recorded at that pixel (the actual ray-hit location, which is
    more accurate than re-deriving X,Y from the grid spacing)."""
    col, row = xy_pixel[0], xy_pixel[1]
    r = int(np.clip(round(row), 0, info['resolution'] - 1))
    c = int(np.clip(round(col), 0, info['resolution'] - 1))
    pt = info['top_hit_xyz'][r, c].copy()
    if np.isnan(pt).any():
        for rad in range(1, 8):
            r0, r1 = max(0, r - rad), min(info['resolution'], r + rad + 1)
            c0, c1 = max(0, c - rad), min(info['resolution'], c + rad + 1)
            patch = info['top_hit_xyz'][r0:r1, c0:c1].reshape(-1, 3)
            valid = patch[~np.isnan(patch).any(axis=1)]
            if len(valid) > 0:
                # nearest valid sample to the requested pixel, in pixel space
                pr, pc = np.mgrid[r0:r1, c0:c1]
                pr = pr.reshape(-1); pc = pc.reshape(-1)
                flat = info['top_hit_xyz'][r0:r1, c0:c1].reshape(-1, 3)
                ok = ~np.isnan(flat).any(axis=1)
                dd = (pr[ok] - row) ** 2 + (pc[ok] - col) ** 2
                pt = flat[ok][np.argmin(dd)]
                break
    return pt


def process_pose(mesh, T, pose_rank, pose_P, hp):
    """Run 3.2.2-3.2.3 for a single stable pose. Returns raw 2D candidate
    dicts (still pixel/posed-frame) plus debug info for visualization."""
    mp = mesh.copy()
    mp.apply_transform(T)
    info = raycast_topdown(mp, resolution=hp['projection']['resolution'],
                            pad_frac=hp['projection']['pad_frac'],
                            top_frac=hp['projection']['top_frac'])
    mask = (info['final_mask'] * 255).astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return [], dict(mask=mask, smooth_outer=None, smooth_holes=[], info=info)
    h = hierarchy[0]
    outer_candidates = [i for i in range(len(contours)) if h[i][3] == -1]
    outer_idx = max(outer_candidates, key=lambda i: cv2.contourArea(contours[i]))
    hole_idx = [i for i in range(len(contours)) if h[i][3] == outer_idx
                and cv2.contourArea(contours[i]) > 4]

    raw_outer = contours[outer_idx][:, 0, :].astype(float)
    smooth_outer, _, _ = fit_efd_contour(raw_outer, order=hp['efd']['order'],
                                          num_points=hp['efd']['num_points'])
    mask_long_dim = float(max(smooth_outer.max(0) - smooth_outer.min(0)))
    diag = float(np.linalg.norm(smooth_outer.max(0) - smooth_outer.min(0)))

    raw_pairs = []
    op = outer_branch_pairs(smooth_outer,
                             min_depth_px=hp['outer_branch']['min_depth_frac'] * diag,
                             max_pairs=hp['outer_branch']['max_pairs_per_pose'])
    for p in op:
        p['mask_long_dim_px'] = mask_long_dim
        p['pose_rank'] = pose_rank
        p['pose_P'] = pose_P
        raw_pairs.append(p)

    smooth_holes = []
    for hi in hole_idx:
        raw_h = contours[hi][:, 0, :].astype(float)
        if len(raw_h) < 5:
            continue
        smooth_h, _, _ = fit_efd_contour(raw_h, order=hp['efd']['order'], num_points=300)
        smooth_holes.append(smooth_h)
        pr = hole_branch_pair(smooth_h, smooth_outer,
                               pixel_size_mm=info['pixel_size'],
                               finger_pad_conform_mm=hp['scoring']['finger_pad_conform_mm'])
        pr['mask_long_dim_px'] = mask_long_dim
        pr['pose_rank'] = pose_rank
        pr['pose_P'] = pose_P
        raw_pairs.append(pr)

    for p in raw_pairs:
        p['xyz1'] = _lookup_xyz(info, p['p1'])
        p['xyz2'] = _lookup_xyz(info, p['p2'])

    debug = dict(mask=mask, smooth_outer=smooth_outer, smooth_holes=smooth_holes, info=info)
    return raw_pairs, debug


def attach_3d_and_score(raw_pairs, T, surfq, com, hp):
    out = []
    for p in raw_pairs:
        p1_3d_posed = p['xyz1']
        p2_3d_posed = p['xyz2']
        p1_obj = unrotate(p1_3d_posed, T)
        p2_obj = unrotate(p2_3d_posed, T)

        frame = build_grasp_frame(p1_obj, p2_obj, surfq, knn_k=hp['approach_vector']['knn_k'])
        cand = dict(p1=p1_obj, p2=p2_obj, pkm=frame['pkm'], u=frame['u'], v=frame['v'],
                    x=frame['x'], n1=frame['n1'], n2=frame['n2'], width=frame['width'],
                    kind=p['kind'], depth_px=p.get('depth_px', np.nan),
                    mask_long_dim_px=p['mask_long_dim_px'],
                    pose_rank=p['pose_rank'], pose_P=p['pose_P'])
        score_candidate(cand, com, hp['scoring_runtime'])
        out.append(cand)
    return out


def dedup_candidates(cands, diag_mm, hp):
    thr_dist = hp['dedup']['center_dist_frac_of_diag'] * diag_mm
    thr_cos = hp['dedup']['axis_cos_threshold']
    cands_sorted = sorted(cands, key=lambda c: -c['Q'])
    kept = []
    for c in cands_sorted:
        is_dup = False
        for k in kept:
            if c['kind'] != k['kind']:
                continue
            d = np.linalg.norm(c['pkm'] - k['pkm'])
            if d < thr_dist and abs(np.dot(c['u'], k['u'])) > thr_cos:
                is_dup = True
                break
        if not is_dup:
            kept.append(c)
    return kept


def run_object(mesh, hp, n_surface_samples=40000):
    com = mesh.center_mass
    diag_mm = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    hp['scoring_runtime'] = dict(
        d0_outer_frac=hp['scoring']['d0_outer_frac'],
        d0_hole_frac=hp['scoring']['d0_hole_frac'],
        mu=hp['scoring']['mu'],
        w_min_mm=hp['scoring']['gripper']['w_min_mm'],
        w_max_mm=hp['scoring']['gripper']['w_max_mm'],
        gamma=hp['scoring']['gripper']['gamma'],
        sigma_w_mm=hp['scoring']['gripper']['sigma_w_mm'],
        sigma_c_mm=hp['scoring']['moment_arm']['sigma_c_frac_of_diag'] * diag_mm,
        lambda_=None)
    hp['scoring_runtime']['lambda'] = hp['scoring']['lambda']

    top_k_poses, all_poses = compute_stable_poses(mesh, top_k=hp['stable_pose']['top_k'])
    surfq = SurfaceQuery(mesh, n_samples=n_surface_samples)

    all_cands = []
    pose_debug = []
    for rank, t in enumerate(top_k_poses):
        raw_pairs, dbg = process_pose(mesh, t['T'], rank, t['P'], hp)
        cands = attach_3d_and_score(raw_pairs, t['T'], surfq, com, hp)
        all_cands.extend(cands)
        pose_debug.append(dict(pose=t, debug=dbg, n_raw=len(raw_pairs)))

    final = dedup_candidates(all_cands, diag_mm, hp)
    final.sort(key=lambda c: -c['Q'])
    return dict(candidates_all=all_cands, candidates_final=final,
                pose_debug=pose_debug, top_k_poses=top_k_poses, com=com, diag_mm=diag_mm)

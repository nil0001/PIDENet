"""Overlay projected grasp candidates on RGB frames for visual sanity check."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .dataset_io import load_gt, load_info, load_phase1_candidates, frame_paths
from .frame_labeling import (transform_point, transform_direction,
                              project_point)


PALETTE_BGR = [(75, 25, 230), (75, 180, 60), (216, 99, 67), (49, 130, 245),
                (180, 30, 145), (240, 240, 70), (230, 50, 240), (12, 247, 188)]


def _draw_grasp_2d(img_bgr: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray,
                    cand: dict, color, S: float, rank: int,
                    approach_len_mm: float = 30.0):
    """Draw the grasp axis (p1-p2) plus the approach vector arrow on img_bgr."""
    p1_cam = transform_point(np.array(cand['contact_p1']), R, t)
    p2_cam = transform_point(np.array(cand['contact_p2']), R, t)
    pkm_cam = transform_point(np.array(cand['center_pkm']), R, t)
    v_cam = transform_direction(np.array(cand['approach_vector_v']), R)

    u1, v1 = project_point(p1_cam, K)
    u2, v2 = project_point(p2_cam, K)
    uk, vk = project_point(pkm_cam, K)
    if any(np.isnan(x) for x in [u1, v1, u2, v2, uk, vk]):
        return

    # grasp closing axis
    cv2.line(img_bgr, (int(u1), int(v1)), (int(u2), int(v2)), color, 2)
    cv2.circle(img_bgr, (int(u1), int(v1)), 3, color, -1)
    cv2.circle(img_bgr, (int(u2), int(v2)), 3, color, -1)

    # approach vector arrow
    tip_cam = pkm_cam + v_cam * approach_len_mm
    ut, vt = project_point(tip_cam, K)
    if not np.isnan(ut):
        cv2.arrowedLine(img_bgr, (int(uk), int(vk)), (int(ut), int(vt)),
                         color, 2, tipLength=0.3)

    cv2.putText(img_bgr, f'#{rank} S={S:.2f}', (int(uk) + 6, int(vk) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def render_frame_overlay(dataset_root: Path, obj_id: int, frame_id: int,
                          candidates: list[dict], labeled_frame: dict,
                          min_S: float = 0.0, top_n: int = 6) -> np.ndarray:
    """Return an RGB image (uint8, HxWx3) with projected grasps drawn."""
    dataset_root = Path(dataset_root)
    p = frame_paths(dataset_root, obj_id, frame_id)
    img = cv2.imread(str(p['rgb']))
    if img is None:
        raise FileNotFoundError(p['rgb'])
    gt = load_gt(dataset_root / 'data' / f'{obj_id:02d}' / 'gt.yml')
    info = load_info(dataset_root / 'data' / f'{obj_id:02d}' / 'info.yml')

    R = gt[frame_id]['R']
    t = gt[frame_id]['t']
    K = info[frame_id]['K']

    # keep the ranking from the labeled frame (already ordered by S)
    poses = list(labeled_frame.values())
    poses = [pp for pp in poses if pp['S'] >= min_S]
    poses.sort(key=lambda pp: -pp['S'])
    poses = poses[:top_n]

    # look up matching candidate by id
    id_to_cand = {c['id']: c for c in candidates}
    for rank, pp in enumerate(poses):
        cand = id_to_cand[pp['id']]
        color = PALETTE_BGR[rank % len(PALETTE_BGR)]
        _draw_grasp_2d(img, K, R, t, cand, color, pp['S'], rank + 1)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def make_sample_grid(dataset_root: Path, obj_id: int, tag: str,
                      candidates: list[dict], labels: dict,
                      out_path: Path, sample_frame_ids: list[int] | None = None,
                      n_samples: int = 6, top_n: int = 5, min_S: float = 0.0,
                      seed: int = 0):
    """Save a 2x3 grid of RGB frames with grasps overlaid, to eyeball
    whether the projection is correct across many views."""
    fids = sample_frame_ids or list(labels.keys())
    rng = np.random.default_rng(seed)
    if len(fids) > n_samples:
        picks = sorted(rng.choice(fids, size=n_samples, replace=False))
    else:
        picks = fids
    ncols = 3
    nrows = (len(picks) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
    axes = np.atleast_2d(axes).flatten()
    for ax, fid in zip(axes, picks):
        img = render_frame_overlay(dataset_root, obj_id, fid,
                                     candidates, labels[fid],
                                     min_S=min_S, top_n=top_n)
        ax.imshow(img)
        ax.set_title(f'frame {fid:04d}', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[len(picks):]:
        ax.axis('off')
    plt.suptitle(f'{tag}  —  Phase-2 projected grasps (top-{top_n} by S; arrow = approach vector v)',
                  fontsize=12, weight='bold', y=1.005)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()

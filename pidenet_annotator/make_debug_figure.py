"""Produce the paper-style multi-panel debug figure that shows every stage
of the annotation pipeline (silhouette → upper-band mask → carved final
mask → EFD contours + dual-branch pairs) for each of the top-4 stable
poses. One figure per object.

Same CLI as run_phase1.py: pass either --linemod ... --objects ... or
--plys explicit paths.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import cv2
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pidenet_annotator import load_hp
from pidenet_annotator.stable_pose import compute_stable_poses
from pidenet_annotator.projection import raycast_topdown
from pidenet_annotator.grasp2d import (fit_efd_contour,
                                        outer_branch_pairs,
                                        hole_branch_pair)


LINEMOD_TAGS = {1: 'ape', 5: 'kettle'}


def draw_stage_row(mesh, hp, pose_entry, pose_rank, row_axes, T):
    mp = mesh.copy()
    mp.apply_transform(T)
    info = raycast_topdown(mp, resolution=hp['projection']['resolution'],
                            pad_frac=hp['projection']['pad_frac'],
                            top_frac=hp['projection']['top_frac'])

    row_axes[0].imshow(info['naive_mask'], origin='lower', cmap='gray_r')
    row_axes[0].set_title(f'pose#{pose_rank+1}  P={pose_entry["P"]*100:.0f}%\n'
                          f'(a) posed & silhouette')
    row_axes[0].set_xticks([]); row_axes[0].set_yticks([])

    row_axes[1].imshow(info['top_mask'], origin='lower', cmap='gray_r')
    row_axes[1].set_title('(b) upper-30% mask')
    row_axes[1].set_xticks([]); row_axes[1].set_yticks([])

    row_axes[2].imshow(info['final_mask'], origin='lower', cmap='gray_r')
    row_axes[2].set_title('(c) final mask\n(carved holes if enclosed)')
    row_axes[2].set_xticks([]); row_axes[2].set_yticks([])

    mask = (info['final_mask'] * 255).astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    h = hierarchy[0] if hierarchy is not None else np.array([])
    ax3 = row_axes[3]
    ax3.imshow(info['final_mask'], origin='lower', cmap='gray_r')
    if len(contours) > 0:
        outer_idx = max([i for i in range(len(contours)) if h[i][3] == -1],
                         key=lambda i: cv2.contourArea(contours[i]))
        hole_idx = [i for i in range(len(contours)) if h[i][3] == outer_idx
                    and cv2.contourArea(contours[i]) > 4]
        raw_outer = contours[outer_idx][:, 0, :].astype(float)
        smooth_outer, _, _ = fit_efd_contour(raw_outer, order=hp['efd']['order'],
                                              num_points=hp['efd']['num_points'])
        diag = float(np.linalg.norm(smooth_outer.max(0) - smooth_outer.min(0)))
        pairs = outer_branch_pairs(smooth_outer,
                                    min_depth_px=hp['outer_branch']['min_depth_frac'] * diag,
                                    max_pairs=hp['outer_branch']['max_pairs_per_pose'])
        ax3.plot(raw_outer[:, 0], raw_outer[:, 1], color='#888', lw=0.5, alpha=0.5)
        ax3.plot(smooth_outer[:, 0], smooth_outer[:, 1], color='#0057b7', lw=1.2, label='EFD outer')
        for pr in pairs:
            ax3.plot([pr['p1'][0], pr['p2'][0]], [pr['p1'][1], pr['p2'][1]],
                     'r-o', lw=1.6, ms=3)
        for hi in hole_idx:
            raw_h = contours[hi][:, 0, :].astype(float)
            if len(raw_h) < 5:
                continue
            smooth_h, _, _ = fit_efd_contour(raw_h, order=hp['efd']['order'], num_points=300)
            ax3.plot(smooth_h[:, 0], smooth_h[:, 1], color='#f0a020', lw=1.2)
            hp_pair = hole_branch_pair(smooth_h, smooth_outer,
                                        pixel_size_mm=info['pixel_size'],
                                        finger_pad_conform_mm=hp['scoring']['finger_pad_conform_mm'])
            ax3.plot([hp_pair['p1'][0], hp_pair['p2'][0]],
                     [hp_pair['p1'][1], hp_pair['p2'][1]],
                     'm-o', lw=1.8, ms=3.5)
    ax3.set_title('(d) EFD + dual-branch pairs\nred=outer defect  magenta=hole')
    ax3.set_xticks([]); ax3.set_yticks([])


def build_pipeline_figure(mesh_file, tag, hp, out_dir: Path):
    mesh = trimesh.load(str(mesh_file), process=False)
    top_k, _ = compute_stable_poses(mesh, top_k=hp['stable_pose']['top_k'])
    K = len(top_k)
    fig, axes = plt.subplots(K, 4, figsize=(14, 3.4 * K))
    if K == 1:
        axes = axes[None, :]
    for k, entry in enumerate(top_k):
        draw_stage_row(mesh, hp, entry, k, axes[k], entry['T'])
    plt.suptitle(f'{tag} — annotation pipeline stages across top-{K} stable poses',
                  y=1.001, fontsize=14, weight='bold')
    plt.tight_layout()
    out = out_dir / f'pipeline_stages_{tag}.png'
    plt.savefig(out, dpi=110, bbox_inches='tight')
    plt.close()
    return out


def resolve_ply_paths(args):
    if args.plys:
        return [(Path(p), Path(p).stem) for p in args.plys]
    if not args.linemod:
        raise SystemExit("Pass --linemod ... --objects ... or --plys ...")
    root = Path(args.linemod).expanduser()
    models_dir = root / 'models'
    out = []
    for obj_id in args.objects:
        candidates = [models_dir / f'obj_{obj_id:02d}.ply', models_dir / f'obj_{obj_id}.ply']
        found = next((c for c in candidates if c.exists()), None)
        if found is None:
            raise SystemExit(f"Could not find PLY for object id {obj_id}. Tried {candidates}")
        tag = LINEMOD_TAGS.get(obj_id, f'obj_{obj_id:02d}')
        out.append((found, tag))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('--linemod', type=str, default=None)
    grp.add_argument('--plys', nargs='+', default=None)
    parser.add_argument('--objects', type=int, nargs='+', default=[1, 5])
    parser.add_argument('--out', type=str, default='outputs')
    parser.add_argument('--hyperparams', type=str, default='hyperparams.yml')
    args = parser.parse_args()

    hp = load_hp(args.hyperparams)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for f, t in resolve_ply_paths(args):
        p = build_pipeline_figure(f, t, hp, out_dir)
        print(f'wrote: {p}')


if __name__ == '__main__':
    main()

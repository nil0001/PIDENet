"""Phase-2 driver: transform Phase-1 candidates into every frame's camera
coordinate system, compute a collision-aware score using self-occlusion
raycasting + segmentation-mask scene-occlusion, and dump per-object YAML
label files.

Usage:
    python run_phase2.py --linemod E:\\paper\\PIDENet\\LINEMOD ^
                         --phase1 outputs ^
                         --out outputs ^
                         --objects 1 5
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import trimesh
import yaml

from pidenet_annotator import load_hp
from pidenet_annotator.phase2 import (label_object_frames,
                                        load_phase1_candidates)
from pidenet_annotator.phase2.frame_labeling import dump_phase2_yaml
from pidenet_annotator.phase2.visualize import make_sample_grid


LINEMOD_TAGS = {1: 'ape', 5: 'kettle'}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--linemod', type=str, required=True,
                        help='Root of the LINEMOD-style dataset '
                             '(must contain models/ and data/).')
    parser.add_argument('--phase1', type=str, required=True,
                        help='Directory containing phase1_candidates_<tag>.yml '
                             '(produced by run_phase1.py).')
    parser.add_argument('--out', type=str, default='outputs',
                        help='Output directory for phase-2 YAMLs and sample grids.')
    parser.add_argument('--objects', type=int, nargs='+', default=[1, 5],
                        help='Object ids to process (default: 1 5).')
    parser.add_argument('--hyperparams', type=str, default='hyperparams.yml')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Debug: cap the number of frames per object.')
    parser.add_argument('--min-S', type=float, default=0.0,
                        help='Optional lower bound on per-frame score S '
                             'for the sample-grid visualization ONLY. The '
                             'YAML file always contains every pose.')
    parser.add_argument('--sample-frames', type=int, default=6,
                        help='How many random frames to render per object '
                             'for the visual sanity check (default: 6).')
    parser.add_argument('--top-n-viz', type=int, default=5,
                        help='How many top candidates to draw per sample frame.')
    args = parser.parse_args()

    hp = load_hp(args.hyperparams)
    # sensible defaults if the hyperparams file doesn't have a phase2 block yet
    hp.setdefault('phase2', {})
    hp['phase2'].setdefault('self_occ_tol_mm', 3.0)
    hp['phase2'].setdefault('scene_occ_margin_px', 2)
    hp['phase2'].setdefault('alpha', 2.0)

    dataset_root = Path(args.linemod).expanduser()
    phase1_dir = Path(args.phase1).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    for obj_id in args.objects:
        tag = LINEMOD_TAGS.get(obj_id, f'obj_{obj_id:02d}')
        print(f'\n=== object {obj_id:02d}  ({tag}) ===')

        phase1_yml = phase1_dir / f'phase1_candidates_{tag}.yml'
        if not phase1_yml.exists():
            print(f'  ERROR: {phase1_yml} not found -- skipping. '
                  f'(Did you run run_phase1.py first?)')
            continue
        cands_doc = load_phase1_candidates(phase1_yml)
        cands = cands_doc['candidates']
        print(f'  loaded {len(cands)} Phase-1 candidates from {phase1_yml.name}')

        # frame list
        gt_path = dataset_root / 'data' / f'{obj_id:02d}' / 'gt.yml'
        with open(gt_path, encoding='utf-8') as f:
            gt_full = yaml.safe_load(f)
        all_fids = sorted(int(k) for k in gt_full.keys())
        if args.max_frames:
            all_fids = all_fids[:args.max_frames]
        print(f'  labeling {len(all_fids)} frames from {gt_path.parent}...')

        t0 = time.time()
        labels = label_object_frames(dataset_root, obj_id, phase1_yml, hp,
                                       frame_ids=all_fids)
        dt = time.time() - t0
        print(f'  done: {len(labels)} labeled frames in {dt:.1f}s '
              f'({dt/max(1,len(labels))*1000:.1f} ms/frame)')

        yml_out = out_dir / f'phase2_labels_{tag}.yml'
        dump_phase2_yaml(labels, yml_out)
        print(f'  wrote: {yml_out}')

        grid_out = out_dir / f'phase2_sample_grid_{tag}.png'
        make_sample_grid(dataset_root, obj_id, tag, cands, labels, grid_out,
                          n_samples=args.sample_frames,
                          top_n=args.top_n_viz, min_S=args.min_S)
        print(f'  wrote: {grid_out}')


if __name__ == '__main__':
    main()

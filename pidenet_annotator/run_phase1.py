"""Phase-1 driver: run the annotation pipeline on each requested object
from a Linemod-style dataset directory.

Usage examples:
    # Windows (from a terminal or double-click):
    python run_phase1.py --linemod E:\\paper\\PIDENet\\LINEMOD --objects 1 5

    # Ubuntu:
    python run_phase1.py --linemod ~/datasets/LINEMOD --objects 1 5

    # Point directly at ply files instead of a dataset dir:
    python run_phase1.py --plys path/to/obj_01.ply path/to/obj_05.ply
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh
import yaml
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go

from pidenet_annotator import load_hp, run_object
from pidenet_annotator import viz3d


PALETTE = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#46f0f0',
           '#f032e6', '#bcf60c', '#fabebe', '#008080', '#9a6324', '#000075']

# Convenience mapping for the two Linemod objects the user asked about
LINEMOD_TAGS = {1: 'ape', 5: 'kettle'}


def dump_phase1_yaml(res, tag, out_path):
    """Object-frame candidates -- what Phase 2 will consume."""
    doc = dict(
        object=tag,
        object_frame_units='mm',
        object_diag_mm=float(res['diag_mm']),
        object_center_of_mass_mm=[float(x) for x in res['com']],
        top_stable_poses=[dict(rank=i + 1, P=float(t['P']))
                           for i, t in enumerate(res['top_k_poses'])],
        candidates=[]
    )
    for i, c in enumerate(res['candidates_final']):
        doc['candidates'].append(dict(
            id=i + 1,
            kind=c['kind'],
            source_pose_rank=int(c['pose_rank']) + 1,
            source_pose_P=float(c['pose_P']),
            width_mm=float(c['width']),
            center_pkm=[float(x) for x in c['pkm']],
            contact_p1=[float(x) for x in c['p1']],
            contact_p2=[float(x) for x in c['p2']],
            approach_vector_v=[float(x) for x in c['v']],
            orientation_vector_u=[float(x) for x in c['u']],
            local_x_axis=[float(x) for x in c['x']],
            Q_score=float(c['Q']),
            feasible=bool(c['feasible']),
            component_scores=dict(
                geo=float(c['S_geo']), ali=float(c['S_ali']),
                wid=float(c['S_wid']), com=float(c['S_com']),
            ),
            friction_angles_deg=[float(c['friction_angle1_deg']),
                                  float(c['friction_angle2_deg'])],
            com_perpendicular_distance_mm=float(c['com_perp_dist_mm']),
        ))
    with open(out_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=None)
    return out_path


def static_multiview(mesh, res, tag, out_path, n_show=6):
    v = mesh.vertices
    f = mesh.faces
    if mesh.visual.kind == 'vertex':
        vc = mesh.visual.vertex_colors[:, :3] / 255.0
        face_colors = vc[f].mean(axis=1)
    else:
        face_colors = np.tile(np.array([0.75, 0.6, 0.55]), (len(f), 1))
    arrow_len = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])) * 0.14

    views = [(15, 30, 'iso 1 (elev15,azim30)'),
             (15, 120, 'iso 2 (elev15,azim120)'),
             (15, 210, 'iso 3 (elev15,azim210)'),
             (90, -90, 'top-down (elev90,azim-90)')]
    fig = plt.figure(figsize=(5.2 * len(views), 5.8))
    for vi, (elev, azim, vname) in enumerate(views):
        ax = fig.add_subplot(1, len(views), vi + 1, projection='3d')
        pc = Poly3DCollection(v[f], facecolor=face_colors, edgecolor='none', alpha=0.4)
        ax.add_collection3d(pc)
        for i, c in enumerate(res['candidates_final'][:n_show]):
            color = PALETTE[i % len(PALETTE)]
            p1, p2, pkm, vv = c['p1'], c['p2'], c['pkm'], c['v']
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                     '-o', color=color, lw=3.2, ms=5,
                     label=(f'#{i+1} {c["kind"]}  Q={c["Q"]:.2f}' if vi == 0 else None))
            vt = pkm + vv * arrow_len
            ax.plot([pkm[0], vt[0]], [pkm[1], vt[1]], [pkm[2], vt[2]],
                     ':', color=color, lw=1.6)
            ax.text(pkm[0], pkm[1], pkm[2], f' #{i+1}',
                     color=color, fontsize=10, weight='bold')
        mx = v.max(0); mn = v.min(0); c0 = (mx + mn) / 2
        r = (mx - mn).max() / 2 * 1.15
        ax.set_xlim(c0[0] - r, c0[0] + r)
        ax.set_ylim(c0[1] - r, c0[1] + r)
        ax.set_zlim(c0[2] - r, c0[2] + r)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('X (mm)', labelpad=0)
        ax.set_ylabel('Y (mm)', labelpad=0)
        ax.set_zlabel('Z (mm)', labelpad=0)
        ax.set_title(vname, fontsize=11)
    fig.legend(loc='lower center', ncol=min(n_show, 6), fontsize=10,
                bbox_to_anchor=(0.5, -0.02))
    plt.suptitle(f'{tag} — top-{n_show} grasp candidates (object frame; dotted = approach vector v)',
                  y=1.0, fontsize=13, weight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()


def interactive_html(mesh, res, tag, out_path, n_show=8):
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    arrow_len = diag * 0.18
    fig = go.Figure()
    fig.add_trace(viz3d.make_mesh_trace(mesh, opacity=0.5))
    for i, c in enumerate(res['candidates_final'][:n_show]):
        color = PALETTE[i % len(PALETTE)]
        for t in viz3d.candidate_traces(c, i + 1, arrow_len, color):
            fig.add_trace(t)
    fig.update_layout(
        title=f'{tag} — top-{n_show} grasp candidates (object canonical frame)',
        height=820, width=1150,
        scene=dict(aspectmode='data',
                   xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)'),
        legend=dict(itemsizing='constant'))
    fig.write_html(out_path)


def resolve_ply_paths(args):
    """Return list of (path, tag) for every .ply we should process."""
    if args.plys:
        return [(Path(p), Path(p).stem) for p in args.plys]

    if not args.linemod:
        raise SystemExit(
            "Neither --linemod nor --plys was given. "
            "Pass either --linemod E:\\paper\\PIDENet\\LINEMOD --objects 1 5, "
            "or --plys path/to/obj_01.ply path/to/obj_05.ply")

    root = Path(args.linemod).expanduser()
    if not root.exists():
        raise SystemExit(f"Linemod path not found: {root}")
    models_dir = root / 'models'
    if not models_dir.exists():
        raise SystemExit(f"'models' subdirectory not found under {root}")

    out = []
    for obj_id in args.objects:
        candidates = [models_dir / f'obj_{obj_id:02d}.ply',
                      models_dir / f'obj_{obj_id}.ply']
        found = next((c for c in candidates if c.exists()), None)
        if found is None:
            raise SystemExit(
                f"Could not find PLY for object id {obj_id} under {models_dir}. "
                f"Tried: {[str(c) for c in candidates]}")
        tag = LINEMOD_TAGS.get(obj_id, f'obj_{obj_id:02d}')
        out.append((found, tag))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('--linemod', type=str, default=None,
                     help='Root of a Linemod-style dataset (contains models/, data/, ...).')
    grp.add_argument('--plys', nargs='+', default=None,
                     help='Explicit list of .ply paths to process (bypasses --linemod).')
    parser.add_argument('--objects', type=int, nargs='+', default=[1, 5],
                        help='Object ids to load from models/ (default: 1 5). '
                             'Ignored when --plys is given.')
    parser.add_argument('--out', type=str, default='outputs',
                        help='Output directory (will be created).')
    parser.add_argument('--hyperparams', type=str, default='hyperparams.yml',
                        help='Path to hyperparams.yml.')
    parser.add_argument('--top-n', type=int, default=8,
                        help='How many top candidates to visualize.')
    args = parser.parse_args()

    hp = load_hp(args.hyperparams)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ply_path, tag in resolve_ply_paths(args):
        print(f'\n=== {tag} ({ply_path}) ===')
        mesh = trimesh.load(str(ply_path), process=False)
        res = run_object(mesh, hp)
        print(f'  candidates final: {len(res["candidates_final"])}')
        for i, c in enumerate(res['candidates_final'][:args.top_n]):
            print(f'   #{i+1} kind={c["kind"]:5s} Q={c["Q"]:.3f} '
                  f'w={c["width"]:5.1f}mm  feasible={c["feasible"]}  pose#{c["pose_rank"]+1}')
        p1 = dump_phase1_yaml(res, tag, out_dir / f'phase1_candidates_{tag}.yml')
        p2 = out_dir / f'preview_final_{tag}.png'
        static_multiview(mesh, res, tag, p2, n_show=args.top_n)
        p3 = out_dir / f'candidates_3d_{tag}.html'
        interactive_html(mesh, res, tag, p3, n_show=args.top_n)
        print(f'  wrote: {p1}\n         {p2}\n         {p3}')

    print(f'\nAll outputs written under: {out_dir.resolve()}')


if __name__ == '__main__':
    main()

"""Dataset I/O for Phase 2: LINEMOD-style gt.yml / info.yml readers, plus
loader for Phase-1 candidate YAML files."""
from __future__ import annotations
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_gt(path: str | Path) -> dict[int, dict]:
    """Return {frame_id: {'R': (3,3), 't': (3,), 'obj_id': int, 'obj_bb': [x,y,w,h]}}.

    Each frame's gt entry is a list of per-object dicts (LINEMOD supports
    multiple objects per frame). For LINEMOD/data/01 and /05 each entry is
    a singleton, so we simply take the first element.
    """
    with open(path, encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    out = {}
    for fid, entries in raw.items():
        entry = entries[0] if isinstance(entries, list) else entries
        out[int(fid)] = dict(
            R=np.array(entry['cam_R_m2c'], dtype=float).reshape(3, 3),
            t=np.array(entry['cam_t_m2c'], dtype=float),
            obj_id=int(entry['obj_id']),
            obj_bb=list(entry.get('obj_bb', [])),
        )
    return out


def load_info(path: str | Path) -> dict[int, dict]:
    """Return {frame_id: {'K': (3,3), 'depth_scale': float}}."""
    with open(path, encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    out = {}
    for fid, entry in raw.items():
        out[int(fid)] = dict(
            K=np.array(entry['cam_K'], dtype=float).reshape(3, 3),
            depth_scale=float(entry.get('depth_scale', 1.0)),
        )
    return out


def load_phase1_candidates(path: str | Path) -> dict:
    """Return the Phase-1 candidate YAML as a dict."""
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def iter_frame_ids(gt: dict) -> list[int]:
    return sorted(gt.keys())


def frame_paths(dataset_root: Path, obj_id: int, frame_id: int):
    """Convenience: build the standard LINEMOD paths for one frame."""
    root = Path(dataset_root) / 'data' / f'{obj_id:02d}'
    n = f'{frame_id:04d}.png'
    return dict(rgb=root / 'rgb' / n,
                mask=root / 'mask' / n,
                depth=root / 'depth' / n)

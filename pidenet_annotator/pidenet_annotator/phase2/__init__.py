"""Phase 2: per-frame projection of Phase-1 candidates into the camera
coordinate system, plus collision-aware scoring against the actual RGB-D
frame data (self-occlusion via mesh raycasting, scene occlusion via
segmentation mask)."""

from .frame_labeling import label_object_frames
from .dataset_io import (load_gt, load_info, load_phase1_candidates,
                         iter_frame_ids)

__all__ = [
    "label_object_frames",
    "load_gt", "load_info", "load_phase1_candidates", "iter_frame_ids",
]

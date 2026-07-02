"""pidenet_annotator: reproduction of PIDENet Section 3.2 offline annotation
pipeline (stable-pose analysis, ray-cast projection, EFD contour fitting,
dual-branch grasp point extraction, approach vector, and quality scoring)."""

from .pipeline import load_hp, run_object

__all__ = ["load_hp", "run_object"]
__version__ = "0.1.0"

"""
3.2.2 Ray-Casting Orthographic Projection
-------------------------------------------
For a mesh already rotated into a stable resting pose (Z up), cast a dense
grid of rays straight down (-Z) from above the model and record every
intersection (multiple_hits=True), giving for every pixel the full set of
hit heights along that column.

naive_mask  : pixel has >=1 hit anywhere            -> the object's true
                                                        outer silhouette
top_mask    : pixel has >=1 hit in the upper
              `top_frac` (default 30%) of the
              object's bounding-box height

DESIGN DECISION (documented): applying "keep only hits in the upper 30%"
literally and globally would also chop the *bottom* off ordinary solid,
unevenly-tall objects (e.g. the ape's arms/feet, whose own local apex
sits well below the head and would then incorrectly read as "missing"),
not just hollow containers. The paper's own motivating example (bowls /
cups) is specifically about a *cap region fully enclosed by a taller
surrounding wall*. We therefore only let the elevation threshold carve a
hole where the "low" region (hit something, but nothing reaches the
upper band) is *topologically enclosed* by foreground that does reach
the upper band -- i.e. it does not touch the true exterior background.
A "low" region touching the silhouette's own outer boundary (like a
foot) is left untouched. This reduces exactly to the paper's described
behaviour for a thin-walled open container (the cap is surrounded by the
rim wall -> carved into a ring) while leaving solid, unevenly-shaped
objects' silhouettes intact.
"""
import numpy as np
from scipy import ndimage


def raycast_topdown(mesh_posed, resolution=512, pad_frac=0.06, top_frac=0.30):
    bmin, bmax = mesh_posed.bounds
    cx, cy = (bmin[0] + bmax[0]) / 2, (bmin[1] + bmax[1]) / 2
    ex, ey = (bmax[0] - bmin[0]), (bmax[1] - bmin[1])
    half = max(ex, ey) / 2 * (1 + 2 * pad_frac)
    xs = np.linspace(cx - half, cx + half, resolution)
    ys = np.linspace(cy - half, cy + half, resolution)
    pixel_size = xs[1] - xs[0]
    xx, yy = np.meshgrid(xs, ys)  # yy rows = image rows
    n = resolution * resolution
    z_top_launch = bmax[2] + 1.0
    origins = np.stack([xx.ravel(), yy.ravel(), np.full(n, z_top_launch)], axis=1)
    dirs = np.tile(np.array([0, 0, -1.0]), (n, 1))

    locs, idx_ray, idx_tri = mesh_posed.ray.intersects_location(
        origins, dirs, multiple_hits=True)

    z_min_obj, z_max_obj = bmin[2], bmax[2]
    z_thresh = z_min_obj + top_frac * (z_max_obj - z_min_obj)

    naive_mask = np.zeros(n, dtype=bool)
    top_mask = np.zeros(n, dtype=bool)
    top_hit_z = np.full(n, -np.inf)
    top_hit_xyz = np.full((n, 3), np.nan)

    if len(idx_ray) > 0:
        order = np.argsort(idx_ray)
        idx_ray_s = idx_ray[order]
        locs_s = locs[order]
        uniq, start = np.unique(idx_ray_s, return_index=True)
        start = np.append(start, len(idx_ray_s))
        for k in range(len(uniq)):
            ri = uniq[k]
            zs = locs_s[start[k]:start[k + 1], 2]
            naive_mask[ri] = True
            if np.any(zs >= z_thresh):
                top_mask[ri] = True
            top_i = np.argmax(zs)
            top_hit_z[ri] = zs[top_i]
            top_hit_xyz[ri] = locs_s[start[k]:start[k + 1]][top_i]

    naive_mask = naive_mask.reshape(resolution, resolution)
    top_mask = top_mask.reshape(resolution, resolution)
    top_hit_xyz = top_hit_xyz.reshape(resolution, resolution, 3)

    # --- enclosed-low-region carving (see module docstring) ---
    low_mask = naive_mask & (~top_mask)
    ext_bg = ~naive_mask
    ext_bg_dil = ndimage.binary_dilation(ext_bg, iterations=1)
    lbl, n_comp = ndimage.label(low_mask, structure=np.ones((3, 3)))
    carve = np.zeros_like(low_mask)
    for comp_id in range(1, n_comp + 1):
        comp = lbl == comp_id
        touches_exterior = np.any(comp & ext_bg_dil)
        if not touches_exterior:
            carve |= comp
    final_mask = naive_mask & (~carve)

    info = dict(xs=xs, ys=ys, pixel_size=pixel_size, cx=cx, cy=cy, half=half,
                z_thresh=z_thresh, z_min_obj=z_min_obj, z_max_obj=z_max_obj,
                naive_mask=naive_mask, top_mask=top_mask, carve_mask=carve,
                final_mask=final_mask, top_hit_xyz=top_hit_xyz,
                resolution=resolution)
    return info


def pixel_to_world(info, row, col):
    """3D point (top-most surface hit) for a given pixel index."""
    return info['top_hit_xyz'][int(round(row)), int(round(col))]


def world_xy_to_pixel(info, x, y):
    col = (x - (info['cx'] - info['half'])) / info['pixel_size']
    row = (y - (info['cy'] - info['half'])) / info['pixel_size']
    return row, col

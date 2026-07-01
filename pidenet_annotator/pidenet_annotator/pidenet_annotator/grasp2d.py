"""
3.2.3 EFD Smooth Curve Reconstruction + dual-branch contact-point search
3.2.4 Approach Vector (KNN + PCA + cross product)
"""
import numpy as np
import cv2
import pyefd
from shapely.geometry import LineString, Polygon, Point
from scipy.spatial.distance import cdist


# ---------- 3.2.3a: EFD fitting ----------
def fit_efd_contour(raw_xy, order=10, num_points=400):
    """raw_xy: (N,2) pixel-space contour (x,y). Returns a smooth, densely
    resampled, analytic (EFD-reconstructed) closed contour of the same
    approximate shape/scale/position."""
    raw_xy = raw_xy.astype(np.float64)
    coeffs = pyefd.elliptic_fourier_descriptors(raw_xy, order=order, normalize=False)
    a0, c0 = pyefd.calculate_dc_coefficients(raw_xy)
    smooth = pyefd.reconstruct_contour(coeffs, locus=(a0, c0), num_points=num_points)
    return smooth, coeffs, (a0, c0)


def _local_tangent(contour, i, span=6):
    n = len(contour)
    p_next = contour[(i + span) % n]
    p_prev = contour[(i - span) % n]
    t = p_next - p_prev
    nrm = np.linalg.norm(t)
    return t / nrm if nrm > 1e-9 else np.array([1.0, 0.0])


def _inward_normal(contour, i, centroid, span=6):
    t = _local_tangent(contour, i, span)
    n1 = np.array([-t[1], t[0]])
    p = contour[i]
    if np.dot(n1, centroid - p) < 0:
        n1 = -n1
    return n1


# ---------- 3.2.3b branch 1: outer contour, convexity-defect pairs ----------
def outer_branch_pairs(smooth_outer, min_depth_px, ray_span=6, max_pairs=8):
    """For each convexity defect on the (EFD-smoothed) outer contour deep
    enough to matter, p_k1 = the defect's farthest (deepest concave) point;
    cast a ray from p_k1 along the local inward normal until it re-crosses
    the contour boundary -> p_k2. Returns list of dicts with p1,p2 in
    pixel-xy, depth_px, and the defect index pair (for debug drawing).
    """
    poly = Polygon(smooth_outer)
    centroid = np.array(poly.centroid.coords[0])
    diag = np.linalg.norm(smooth_outer.max(axis=0) - smooth_outer.min(axis=0))

    pts_i = np.round(smooth_outer).astype(np.int32).reshape(-1, 1, 2)
    hull_idx = cv2.convexHull(pts_i, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 3:
        return []
    try:
        defects = cv2.convexityDefects(pts_i, hull_idx)
    except cv2.error:
        return []
    if defects is None:
        return []

    results = []
    for d in defects[:, 0, :]:
        s, e, f, depth = d
        depth_px = depth / 256.0
        if depth_px < min_depth_px:
            continue
        p1 = smooth_outer[f]
        normal = _inward_normal(smooth_outer, f, centroid, span=ray_span)
        ray = LineString([p1, p1 + normal * diag * 2])
        boundary = poly.exterior
        inter = ray.intersection(boundary)
        cand = []
        if inter.is_empty:
            continue
        if inter.geom_type == 'Point':
            cand = [np.array(inter.coords[0])]
        elif inter.geom_type == 'MultiPoint':
            cand = [np.array(p.coords[0]) for p in inter.geoms]
        elif inter.geom_type in ('LineString', 'GeometryCollection'):
            for g in getattr(inter, 'geoms', [inter]):
                if hasattr(g, 'coords') and len(g.coords) > 0:
                    cand.append(np.array(g.coords[0]))
        cand = [c for c in cand if np.linalg.norm(c - p1) > 1e-3]
        if not cand:
            continue
        dists = [np.linalg.norm(c - p1) for c in cand]
        p2 = cand[int(np.argmin(dists))]
        results.append(dict(p1=p1, p2=p2, depth_px=float(depth_px),
                             width_px=float(np.linalg.norm(p2 - p1)),
                             kind='outer', defect=(int(s), int(e), int(f))))
    results.sort(key=lambda r: -r['depth_px'])
    return results[:max_pairs]


# ---------- 3.2.3b branch 2: internal hole, nearest cross-contour pair ----------
def hole_branch_pair(smooth_hole, smooth_outer, pixel_size_mm, finger_pad_conform_mm):
    """Globally-shortest connection between a hole's boundary and the outer
    boundary -> pinches across the thinnest wall (e.g. handle thickness).

    For Eq.4's handle-hole geometric-interlocking term we need a "d" that
    captures how strongly the gripper wraps/interlocks around this wall.
    DESIGN ITERATION (documented): we first tried "arc length of the hole
    boundary within a fixed finger-contact chord radius" -- empirically
    this came out nearly identical (~44-45px) for both handles on the
    kettle, since at an 8mm reach both loops are already locally close to
    flat relative to that radius, so it failed to discriminate them at
    all. We replaced it with a direct, monotonic "pad conformance" term:
    a compliant fingertip pad of nominal reach `finger_pad_conform_mm`
    wraps around the *remaining* thinness of the wall once closed --
    d = max(eps, finger_pad_conform_mm - width). A thinner handle wall
    leaves more pad free to conform/wrap around it (stronger topological
    interlock); a wall at or above the pad's conformance reach contributes
    ~0. This is consistent with the paper's stated intent that thinner,
    through-hole walls should be favored, and -- unlike the arc-length
    attempt -- actually varies across our two handles (9-18mm) instead of
    saturating.
    """
    D = cdist(smooth_hole, smooth_outer)
    a, b = np.unravel_index(np.argmin(D), D.shape)
    p1 = smooth_hole[a]
    p2 = smooth_outer[b]
    width_mm = float(D[a, b] * pixel_size_mm)
    d_mm = max(0.5, finger_pad_conform_mm - width_mm)
    d_px = d_mm / pixel_size_mm

    return dict(p1=p1, p2=p2, width_px=float(D[a, b]), kind='hole', depth_px=float(d_px))

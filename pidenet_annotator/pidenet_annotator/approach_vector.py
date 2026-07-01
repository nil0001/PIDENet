"""
3.2.4 Approach Vector
----------------------
Given corrected 2D grasp points (p_k1,p_k2) in a pose's top-down pixel
frame, back-project to 3D (in that pose's frame), then un-rotate into the
object's own canonical coordinate frame (the frame the PLY file / gt.yml
R,t convention uses). Build the local grasp frame:
    u (orientation/rotation vector) = normalize(p_k2 - p_km)
    z_raw = PCA-smallest-variance axis of a KNN surface neighborhood
            around p_km
    v (approach vector) = normalize(z_raw orthogonalized against u)
    x = u x v   (completes a right-handed orthonormal frame; stored for
                 reference / visualization, mirrors how the network's
                 regressed u,v are completed into a full rotation via
                 Gram-Schmidt in Sec. 3.3)

Sign convention (documented choice): v is oriented to point away from
the object's local surface (outward, roughly bisecting the two contact
normals n1_hat+n2_hat) so that "approach" = motion along -v, matching
Fig. 2(b) where v is drawn pointing from the gripper body down into the
grasp -- i.e. the gripper travels along -v to reach the grasp and lifts
out along +v.
"""
import numpy as np
from scipy.spatial import cKDTree


class SurfaceQuery:
    """Pre-built dense surface sample + KD-tree + nearest-surface-normal
    query, all expressed in the object's ORIGINAL canonical frame."""

    def __init__(self, mesh, n_samples=40000, seed=0):
        pts, face_idx = mesh.sample(n_samples, return_index=True)
        self.points = pts
        self.normals = mesh.face_normals[face_idx]
        self.tree = cKDTree(pts)
        self.mesh = mesh

    def knn(self, p, k):
        d, idx = self.tree.query(p, k=k)
        return self.points[idx]

    def nearest_normal(self, p):
        d, idx = self.tree.query(p, k=1)
        return self.normals[idx], self.points[idx]


def unrotate(points_xyz, T):
    """points_xyz: (...,3) in the POSED frame -> object's canonical frame."""
    Tinv = np.linalg.inv(T)
    pts = np.atleast_2d(points_xyz)
    homo = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    out = (Tinv @ homo.T).T[:, :3]
    return out.reshape(np.array(points_xyz).shape)


def build_grasp_frame(p1_obj, p2_obj, surfq: SurfaceQuery, knn_k=40):
    pkm = (p1_obj + p2_obj) / 2.0
    u = p2_obj - p1_obj
    u = u / (np.linalg.norm(u) + 1e-12)

    neigh = surfq.knn(pkm, knn_k)
    neigh_c = neigh - neigh.mean(axis=0)
    cov = neigh_c.T @ neigh_c
    eigval, eigvec = np.linalg.eigh(cov)  # ascending
    z_raw = eigvec[:, 0]  # smallest-variance axis

    # orthogonalize against u (Gram-Schmidt)
    z = z_raw - np.dot(z_raw, u) * u
    nz = np.linalg.norm(z)
    if nz < 1e-8:
        # degenerate (neighborhood ~collinear with u); fall back to any
        # vector orthogonal to u
        tmp = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
        z = tmp - np.dot(tmp, u) * u
        nz = np.linalg.norm(z)
    v = z / nz

    n1, _ = surfq.nearest_normal(p1_obj)
    n2, _ = surfq.nearest_normal(p2_obj)
    n1 = n1[0] if n1.ndim > 1 else n1
    n2 = n2[0] if n2.ndim > 1 else n2
    bisector = n1 + n2
    if np.linalg.norm(bisector) < 1e-6:
        bisector = n1
    if np.dot(v, bisector) < 0:
        v = -v

    x_axis = np.cross(u, v)
    nx = np.linalg.norm(x_axis)
    if nx > 1e-8:
        x_axis = x_axis / nx

    return dict(pkm=pkm, u=u, v=v, x=x_axis, n1=n1, n2=n2,
                width=float(np.linalg.norm(p2_obj - p1_obj)))

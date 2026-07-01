"""
3.2.5 Grasp Quality Scoring (offline, per-candidate, object-only -- the
depth-consistency-aware soft collision term Q~(gk) is intentionally
deferred to the per-frame projection stage, see chat discussion: it is
frame/scene dependent and has nothing to compute against in this
object-only phase).
"""
import numpy as np


def s_geo(d, d0):
    return float(np.tanh(d / d0))


def s_ali(u, n1, n2):
    return float(0.5 * (abs(np.dot(u, n1)) + abs(np.dot(u, n2))))


def s_wid(w, w_max, gamma, sigma_w):
    w_star = gamma * w_max
    return float(np.exp(-((w - w_star) ** 2) / (2 * sigma_w ** 2)))


def s_com(perp_dist, sigma_c):
    return float(np.exp(-(perp_dist ** 2) / (2 * sigma_c ** 2)))


def perpendicular_dist_to_line(point, line_point, line_dir):
    w = point - line_point
    proj = np.dot(w, line_dir) * line_dir
    return float(np.linalg.norm(w - proj))


def friction_feasible(u, n1, n2, mu):
    half_angle = np.arctan(mu)
    a1 = np.arccos(np.clip(abs(np.dot(u, n1)), 0, 1))
    a2 = np.arccos(np.clip(abs(np.dot(u, n2)), 0, 1))
    return bool(a1 < half_angle and a2 < half_angle), float(np.degrees(a1)), float(np.degrees(a2))


def score_candidate(cand, com, hp):
    """cand: dict with u,v,pkm,n1,n2,width(mm),p1,p2,kind('outer'/'hole'),
             depth_px, mask_long_dim_px
       com : object's center of mass (3,) mm
       hp  : hyperparameter dict (see hyperparams.yml)
    returns cand with score fields filled in-place and returned.
    """
    if cand['kind'] == 'outer':
        d0 = hp['d0_outer_frac'] * cand['mask_long_dim_px']
    else:
        d0 = hp['d0_hole_frac'] * cand['mask_long_dim_px']
    Sgeo = s_geo(cand['depth_px'], d0)

    Sali = s_ali(cand['u'], cand['n1'], cand['n2'])

    Swid = s_wid(cand['width'], hp['w_max_mm'], hp['gamma'], hp['sigma_w_mm'])

    perp = perpendicular_dist_to_line(com, cand['pkm'], cand['u'])
    Scom = s_com(perp, hp['sigma_c_mm'])

    feasible, ang1, ang2 = friction_feasible(cand['u'], cand['n1'], cand['n2'], hp['mu'])
    width_ok = hp['w_min_mm'] <= cand['width'] <= hp['w_max_mm']
    I = 1.0 if (feasible and width_ok) else 0.0

    lam = hp['lambda']
    raw = lam['geo'] * Sgeo + lam['ali'] * Sali + lam['wid'] * Swid + lam['com'] * Scom
    Q = I * raw

    cand.update(S_geo=Sgeo, S_ali=Sali, S_wid=Swid, S_com=Scom,
                feasible=bool(I), friction_angle1_deg=ang1, friction_angle2_deg=ang2,
                com_perp_dist_mm=perp, Q=float(Q), d0_px=float(d0))
    return cand

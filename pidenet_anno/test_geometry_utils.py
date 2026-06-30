# -*- coding: utf-8 -*-
"""
test_geometry_utils.py — 验证 geometry_utils.py 中数学公式的正确性。
"""
import sys
sys.path.insert(0, ".")
import numpy as np
from utils.geometry_utils import (
    triangle_solid_angle, polygon_solid_angle_from_apex,
    pca_normal_estimation, gram_schmidt_orthogonalize,
    within_friction_cone, point_to_line_distance
)


def test_solid_angle_full_sphere():
    """已知：一个点被一个完全包围它的闭合曲面所张的总立体角应为 4π (全空间)。
    用一个标准的正八面体（顶点在apex周围对称分布）来验证，
    八面体共8个三角面，每个面对中心点的立体角应该相等，求和应接近4π。
    """
    apex = np.array([0.0, 0.0, 0.0])
    # 正八面体顶点：±1 沿三个坐标轴
    verts = {
        '+x': np.array([1, 0, 0]), '-x': np.array([-1, 0, 0]),
        '+y': np.array([0, 1, 0]), '-y': np.array([0, -1, 0]),
        '+z': np.array([0, 0, 1]), '-z': np.array([0, 0, -1]),
    }
    faces = [
        ('+x', '+y', '+z'), ('+y', '-x', '+z'), ('-x', '-y', '+z'), ('-y', '+x', '+z'),
        ('+x', '-z', '+y'), ('+y', '-z', '-x'), ('-x', '-z', '-y'), ('-y', '-z', '+x'),
    ]
    total = 0.0
    for f in faces:
        v0, v1, v2 = verts[f[0]], verts[f[1]], verts[f[2]]
        total += triangle_solid_angle(apex, v0, v1, v2)

    expected = 4 * np.pi
    print(f"[test_solid_angle_full_sphere] 八面体总立体角={total:.6f}, 期望={expected:.6f}, "
          f"误差={abs(total-expected):.6f}")
    assert abs(total - expected) < 1e-3, "立体角公式验证失败！"
    print("  PASS\n")


def test_solid_angle_known_case():
    """已知：从原点看一个垫在z=1平面上的、边长为2的正方形(中心对准原点正上方)，
    其立体角应满足解析公式 Ω = 4*arcsin(a²/(a²+4h²)) 的相关变体。
    这里用更简单的方式验证：对一个非常小的面元，立体角应该约等于 面积*cos(theta)/r²
    （微分立体角公式），取面元边长0.001，与apex距离为1，法向正对apex。
    """
    apex = np.array([0.0, 0.0, 0.0])
    h = 1.0
    s = 0.001  # 很小的正方形，可用微分近似
    v0 = np.array([-s/2, -s/2, h])
    v1 = np.array([ s/2, -s/2, h])
    v2 = np.array([ s/2,  s/2, h])
    v3 = np.array([-s/2,  s/2, h])

    omega_tri1 = triangle_solid_angle(apex, v0, v1, v2)
    omega_tri2 = triangle_solid_angle(apex, v0, v2, v3)
    omega_total = omega_tri1 + omega_tri2

    area = s * s
    expected = area * 1.0 / (h ** 2)  # cos(theta)=1 因为法向正对apex

    print(f"[test_solid_angle_known_case] 数值立体角={omega_total:.8e}, 微分近似期望={expected:.8e}")
    rel_err = abs(omega_total - expected) / expected
    print(f"  相对误差={rel_err:.6f}")
    assert rel_err < 0.01, "微分立体角近似验证失败！"
    print("  PASS\n")


def test_pca_normal_flat_plane():
    """在 z=0 平面上撒点（带微小噪声模拟真实点云），PCA估计的法向应接近 [0,0,1] 或 [0,0,-1]"""
    np.random.seed(0)
    n = 200
    xy = np.random.uniform(-1, 1, size=(n, 2))
    z = np.random.normal(0, 1e-4, size=n)  # 极小噪声
    points = np.column_stack([xy, z])

    normal = pca_normal_estimation(points, reference_outward=np.array([0, 0, 1]))
    print(f"[test_pca_normal_flat_plane] 估计法向={normal}")
    assert np.allclose(normal, [0, 0, 1], atol=1e-2), "平面PCA法向估计失败！"
    print("  PASS\n")


def test_pca_normal_sign_correction():
    """验证 reference_outward 能正确翻转法向符号"""
    np.random.seed(1)
    n = 100
    xy = np.random.uniform(-1, 1, size=(n, 2))
    z = np.random.normal(0, 1e-4, size=n)
    points = np.column_stack([xy, z])

    normal_pos = pca_normal_estimation(points, reference_outward=np.array([0, 0, 1]))
    normal_neg = pca_normal_estimation(points, reference_outward=np.array([0, 0, -1]))

    print(f"[test_pca_normal_sign_correction] +ref => {normal_pos}, -ref => {normal_neg}")
    assert np.allclose(normal_pos, -normal_neg, atol=1e-6), "符号修正逻辑错误！"
    print("  PASS\n")


def test_gram_schmidt_orthogonality():
    """验证Gram-Schmidt输出严格正交且为单位向量"""
    v_approach = np.array([1.0, 0.3, 0.1])
    u_orientation = np.array([0.2, 1.0, -0.1])  # 故意不垂直

    v, u, x = gram_schmidt_orthogonalize(v_approach, u_orientation)

    print(f"[test_gram_schmidt_orthogonality] v={v}, u={u}, x={x}")
    print(f"  |v|={np.linalg.norm(v):.6f}, |u|={np.linalg.norm(u):.6f}, |x|={np.linalg.norm(x):.6f}")
    print(f"  v·u={np.dot(v,u):.2e}, v·x={np.dot(v,x):.2e}, u·x={np.dot(u,x):.2e}")

    assert abs(np.linalg.norm(v) - 1) < 1e-9
    assert abs(np.linalg.norm(u) - 1) < 1e-9
    assert abs(np.linalg.norm(x) - 1) < 1e-9
    assert abs(np.dot(v, u)) < 1e-9
    assert abs(np.dot(v, x)) < 1e-9
    assert abs(np.dot(u, x)) < 1e-9

    # 验证右手系: x = u × v (注意此处约定 v=z轴approach, u=y轴orientation, x = y×z 标准右手系)
    cross_check = np.cross(u, v)
    print(f"  u×v={cross_check}, 应等于x={x}")
    assert np.allclose(cross_check, x, atol=1e-9), "右手系构造错误！"
    print("  PASS\n")


def test_friction_cone_logic():
    """
    验证修正后的摩擦锥逻辑：
    - grasp_axis 指向物体内部（与outward_normal反向）时，应该判定为within（小角度）
    - grasp_axis 指向物体外部（与outward_normal同向）时，应该判定为not within（180度，远超锥角）
    """
    outward_normal = np.array([0, 0, 1.0])
    mu = 0.5  # cone half angle = arctan(0.5) ≈ 26.57°

    axis_inward = np.array([0, 0, -1.0])  # 完全指向内部，夹角0° < 26.57°
    axis_outward = np.array([0, 0, 1.0])  # 完全指向外部，夹角180° > 26.57°
    axis_perpendicular = np.array([1.0, 0, 0])  # 垂直，夹角90° > 26.57°

    r1 = within_friction_cone(axis_inward, outward_normal, mu)
    r2 = within_friction_cone(axis_outward, outward_normal, mu)
    r3 = within_friction_cone(axis_perpendicular, outward_normal, mu)

    print(f"[test_friction_cone_logic] inward={r1}(期望True), outward={r2}(期望False), "
          f"perpendicular={r3}(期望False)")
    assert r1 == True, "指向内部的闭合轴应判定为合法摩擦锥范围内！"
    assert r2 == False, "指向外部的闭合轴应判定为不合法！"
    assert r3 == False, "垂直方向应超出摩擦锥！"
    print("  PASS\n")


def test_point_to_line_distance():
    point = np.array([1.0, 1.0, 0.0])
    line_point = np.array([0.0, 0.0, 0.0])
    line_dir = np.array([1.0, 0.0, 0.0])  # x轴
    d = point_to_line_distance(point, line_point, line_dir)
    print(f"[test_point_to_line_distance] 点(1,1,0)到x轴的距离={d:.6f}, 期望=1.0")
    assert abs(d - 1.0) < 1e-9
    print("  PASS\n")


if __name__ == "__main__":
    test_solid_angle_full_sphere()
    test_solid_angle_known_case()
    test_pca_normal_flat_plane()
    test_pca_normal_sign_correction()
    test_gram_schmidt_orthogonality()
    test_friction_cone_logic()
    test_point_to_line_distance()
    print("=" * 60)
    print("所有 geometry_utils 单元测试通过！")

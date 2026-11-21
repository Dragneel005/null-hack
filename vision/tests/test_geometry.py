from vision.geometry import angle


def test_right_angle_is_90():
    assert abs(angle((1, 0), (0, 0), (0, 1)) - 90) < 1e-6


def test_collinear_points_are_180():
    assert abs(angle((-1, 0), (0, 0), (2, 0)) - 180) < 1e-6


def test_coincident_points_do_not_crash():
    a = angle((0, 0), (0, 0), (1, 0))
    assert a == a  # not NaN

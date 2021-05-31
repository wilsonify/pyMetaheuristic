from pyMetaheuristic import rando


def test_rando():
    x = rando()
    assert x < 1
    assert x > 0

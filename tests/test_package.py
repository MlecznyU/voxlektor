import voxlektor


def test_version_is_exposed_without_side_effects() -> None:
    assert voxlektor.__version__ == "0.1.0"

from melusine.utils import show_versions


def test_show_versions():
    show_versions()
    assert True


def fake_test():
    assert True is False

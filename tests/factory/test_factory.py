import pytest
factory = pytest.importorskip("factory")


class FakeFactory(factory.Factory):
    fake_attribute = 2

    class Meta:
        model = dict


def test_factory():
    b = FakeFactory()
    assert b["fake_attribute"] == 2

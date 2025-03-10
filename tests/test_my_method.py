import pytest
from mlmmint.my_method import TheBestClass


def test_my_method():
    obj = TheBestClass()
    assert obj.my_method() == "Hello World"

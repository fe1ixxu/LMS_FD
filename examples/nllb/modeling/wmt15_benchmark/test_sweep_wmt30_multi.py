import pytest
from sweep_wmt30_multi import get_predefined_grid, register_grid

from fb_sweep.sweep import hyperparam


def test_register_grid() -> None:
    grids = {}
    g1 = [hyperparam("--foo", 23)]
    g2 = [hyperparam("--bar", 42)]

    @register_grid("one", grids=grids)
    def one():
        return g1

    @register_grid("two", grids=grids)
    def two():
        return g2

    assert get_predefined_grid("one", grids=grids) is g1
    assert get_predefined_grid("two", grids=grids) is g2
    with pytest.raises(KeyError):
        get_predefined_grid("three", grids=grids)


def test_register_grid_duplicate_name() -> None:
    grids = {}
    g1 = [hyperparam("--foo", 23)]
    g2 = []

    @register_grid("one", grids=grids)
    def one():
        return g1

    with pytest.raises(ValueError):

        @register_grid("one", grids=grids)
        def two():
            return g2

    assert get_predefined_grid("one", grids=grids) is g1


def test_register_grid_stacked() -> None:
    grids = {}
    g = [hyperparam("--foo", 23)]

    @register_grid("one", grids=grids)
    @register_grid("two", grids=grids)
    def stacked():
        return g

    assert get_predefined_grid("one", grids=grids) is g
    assert get_predefined_grid("two", grids=grids) is g

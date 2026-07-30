from gamla import functional_generic


def _inc(x):
    return x + 1


def _double(x):
    return x * 2


def test_compose_left_names_result_by_default():
    composed = functional_generic.compose_left(_inc, _double)
    assert composed(3) == 8
    assert composed.__code__.co_filename == __file__


def test_compose_left_skips_naming_when_disabled(monkeypatch):
    monkeypatch.setattr(functional_generic, "_NAME_COMPOSED_FUNCTIONS", False)
    composed = functional_generic.compose_left(_inc, _double)
    assert composed(3) == 8
    assert composed.__code__.co_filename != __file__


def test_compose_result_matches_regardless_of_naming_flag(monkeypatch):
    named = functional_generic.compose(_double, _inc)
    monkeypatch.setattr(functional_generic, "_NAME_COMPOSED_FUNCTIONS", False)
    unnamed = functional_generic.compose(_double, _inc)
    assert named(5) == unnamed(5) == 12

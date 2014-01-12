from sympy import (Symbol, infinite_discontinuties, exp, log, sin, S)

from sympy.utilities.pytest import raises, XFAIL


def test_infinite_disconituties():
    x = Symbol('x', real=True)

    assert infinite_discontinuties(exp(x), x) == []
    assert infinite_discontinuties((1 + x**2)/(x - 1), x) == [1]
    assert infinite_discontinuties(log((x - 2)**2), x) == [2]
    assert infinite_discontinuties(exp(1/(x-2)**2), x) == [2]
    assert infinite_discontinuties(exp(-1/(x-2)**2), x) == []
    assert infinite_discontinuties(exp(exp(1/x)), x) == [0]
    assert infinite_discontinuties(exp(-exp(1/x)), x) == []
    assert infinite_discontinuties(1/(x - sin(2)), x) == [sin(2)]

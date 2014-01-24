from sympy import (Symbol, infinite_discontinuties, exp, log, sin, S)
from sympy.calculus.singularities import _has_only_supported_func

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

    #test if the complex solutions are not returned
    assert infinite_discontinuties(log(x**2 + 1), x) == []


def test__has_only_supported_func():
    x = Symbol('x')

    assert _has_only_supported_func(exp(x), x) is True
    assert _has_only_supported_func(log(x), x) is True
    assert _has_only_supported_func(sin(x), x) is False
    assert _has_only_supported_func(log((x-2)**2) + x**2, x) is True
    assert _has_only_supported_func(log(x)**sin(x) + 1, x) is False
    assert _has_only_supported_func(log(x)**(exp(x) + log(x)) + 1, x) is True
    assert _has_only_supported_func(exp(1/exp(1/(x-2)**2 - 1)), x) is True
    assert _has_only_supported_func(log(exp(1/(x-2)**2) + 1)
                                    + 1/(x+1), x) is True
    assert _has_only_supported_func(log(exp(sin(x) + 1) + 1) + 1, x) is False

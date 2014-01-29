from sympy import (Symbol, infinite_discontinuties, exp, log, sin, S)
from sympy.calculus.singularities import _has_only_supported_func

from sympy.utilities.pytest import raises, XFAIL


def test_infinite_disconituties():
    x = Symbol('x', real=True)

    assert infinite_discontinuties(exp(x), x) == []
    assert infinite_discontinuties(log(x), x) == [0]

    # rational functions
    assert infinite_discontinuties(1/x, x) == [0]
    assert infinite_discontinuties((1 + x**2)/(x - 1), x) == [1]
    assert infinite_discontinuties(1/(x**2 - 3*x + 2), x) == [1, 2]

    # test there are no dublicate solutions
    assert infinite_discontinuties(1/(x**2 + 2*x + 1), x) == [-1]

    # test for non rational coefficients
    assert infinite_discontinuties(1/(x - sin(2)), x) == [sin(2)]

    # exponentials
    assert infinite_discontinuties(exp(1/(x - 2)**2), x) == [2]
    assert infinite_discontinuties(exp(-1/(x - 2)**2), x) == []

    # nested exponentials
    assert infinite_discontinuties(exp(exp(1/x)), x) == [0]
    assert infinite_discontinuties(exp(-exp(1/x)), x) == []

    # exp(...)*exp(...)
    assert infinite_discontinuties(exp(1/(x - 1))*exp(1/(x - 2)**2) + 1, x) == \
        [1, 2]
    assert infinite_discontinuties(exp(exp(1/x))*exp(1/(x-1)), x) == [0, 1]

    # logarithmic
    assert infinite_discontinuties(log((x - 2)**2), x) == [2]
    assert infinite_discontinuties(log(x**2 + 1), x) == []
    assert infinite_discontinuties(log(1/(x - 2)), x) == [2]

    # nested logartihmic
    assert infinite_discontinuties(log(log(1/x**2)), x) == \
        [-1, 0, 1]

    # log(...)*log(...)
    assert infinite_discontinuties(log(1/(x - 1)**2)*log(1/(x - 2)**2), x) == \
        [1, 2]

    # f(x) + g(x)
    assert infinite_discontinuties(exp(1/(x - 1)) + exp(1/x), x) == [0, 1]
    assert infinite_discontinuties(exp(1/(x - 1)) + log((x - 2)**2), x) == \
        [1, 2]
    assert infinite_discontinuties(exp(1/(x-1)) + 1/x, x) == [0, 1]
    assert infinite_discontinuties(log((x-1)**2) + 1/x, x) == [0, 1]


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

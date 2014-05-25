from sympy import (
    Abs, And, Derivative, Dummy, Eq, Float, Function, Gt, I, Integral,
    LambertW, Lt, Matrix, Or, Piecewise, Poly, Q, Rational, S, Symbol,
    Wild, acos, asin, atan, atanh, cos, cosh, diff, erf, erfinv, erfc,
    erfcinv, erf2, erf2inv, exp, expand, im, log, pi, re, sec, sin,
    sinh, sqrt, sstr, symbols, sympify, tan, tanh,
    root, simplify, atan2, arg, Mul, SparseMatrix, ask)

from sympy.core.function import nfloat
from sympy.solvers.solvers import _invert, unrad, checksol, posify, _ispow, \
    det_quick, det_perm, det_minor

from sympy.polys.rootoftools import RootOf

from sympy.utilities.pytest import slow, XFAIL, raises, skip
from sympy.utilities.randtest import test_numerically as tn

from sympy.abc import a, b, c, d, k, h, p, x, y, z, t, q, m

from sympy.solvers.solve_univariate import solve_univariate, invert, \
    solve_as_poly

# TODO: fix the pep8 error in the solvers code and the test
# They are irritating and it is tempting to solve them along with writing the
# code


def test_invert():
    assert invert(x + 3, x) == [x - 3]
    assert invert(x * 3, x) == [x / 3]

    assert invert(exp(x), x) == [log(x)]
    assert invert(exp(3 * x), x) == [log(x) / 3]
    assert invert(exp(x + 3), x) == [log(x) - 3]

    assert invert(exp(x) + 3, x) == [log(x - 3)]
    assert invert(exp(x)*3, x) == [log(x / 3)]

    assert invert(log(x), x) == [exp(x)]
    assert invert(log(3 * x), x) == [exp(x) / 3]
    assert invert(log(x + 3), x) == [exp(x) - 3]

    assert invert(Abs(x), x) == [-x, x]


@XFAIL
def test_invert_lambert():
    assert invert(x * exp(x), x) == LambertW(x)


def test_polynomial():
    assert solve_univariate(3 * x - 2, x) == [Rational(2, 3)]

    assert set(solve_univariate(x ** 2 - 1, x)) == set([-S(1), S(1)])

    assert solve_univariate(x - y ** 3, x) == [y ** 3]
    assert set(solve_univariate(x - y ** 3, y)) == set([
        (-x ** Rational(1, 3)) / 2 + I * sqrt(3) * x ** Rational(1, 3) / 2,
        x ** Rational(1, 3),
        (-x ** Rational(1, 3)) / 2 - I * sqrt(3) * x ** Rational(1, 3) / 2,
    ])

    a11, a12, a21, a22, b1, b2 = symbols('a11,a12,a21,a22,b1,b2')

    assert set(solve_univariate(x ** 3 - 15 * x - 4, x)) == set([
        -2 + 3 ** Rational(1, 2),
        S(4),
        -2 - 3 ** Rational(1, 2)
    ])

    assert set(solve_univariate((x ** 2 - 1) ** 2 - a, x)) == \
        set([sqrt(1 + sqrt(a)), -sqrt(1 + sqrt(a)),
             sqrt(1 - sqrt(a)), -sqrt(1 - sqrt(a))])


def test_solve_rational():
    assert solve_univariate(1/x + 1, x) == [-S.One]


@XFAIL
def test_no_sol():
    assert solve_univariate(4, x) == []
    assert solve_univariate(1/x, x) == []
    assert solve_univariate(-(1 + x)/(2 + x)**2 + 1/(2 + x), x) == []
    assert solve_univariate(-x**2 - 2*x + (x + 1)**2 - 1, x) == []
    assert solve_univariate((x/(x + 1) + 3)**(-2), x) == []
    assert solve_univariate((x - 1)/(1 + 1/(x - 1)), x) == []


def test_solve_polynomial_cv_1a():
    """
    Test for solving on equations that can be converted to
    a polynomial equation using the change of variable y -> x**Rational(p, q)
    """
    assert solve_as_poly(sqrt(x) - 1, x) == [1]
    assert solve_as_poly(sqrt(x) - 2, x) == [4]
    assert solve_as_poly(x ** Rational(1, 4) - 2, x) == [16]
    assert solve_as_poly(x ** Rational(1, 3) - 3, x) == [27]


@XFAIL
def test_solve_polynomial_multiple_gens():
    """
    Test for solving on equations that can be converted to a polynomial
    equation multiplying both sides of the equation by x**m
    """
    assert solve_as_poly(sqrt(x) + x ** Rational(1, 3) +
                         x ** Rational(1, 4), x) == [0]
    assert solve_as_poly(x + 1 / x - 1, x) in \
        [[Rational(1, 2) + I * sqrt(3) / 2, Rational(1, 2) - I * sqrt(3) / 2],
         [Rational(1, 2) - I * sqrt(3) / 2, Rational(1, 2) + I * sqrt(3) / 2]]


@XFAIL
def test_solve_polnomoial_irration_deg():
    assert solve_as_poly(x ** pi - 1, x)


def test_solve_polynomial_symbolic_param():
    assert set(
        solve_as_poly(4 * x * (1 - a * sqrt(x)), x)) == set([S(0), 1 / a ** 2])


def test_solve_polynomial_cv_1b():
    assert set(
        solve_as_poly(x * (x ** (S(1) / 3) - 3), x)) == set([S(0), S(27)])


@XFAIL
def test_highorder_poly():
    sol = solve_univariate(x ** 6 - 2 * x + 2, x)
    assert all(isinstance(i, RootOf) for i in sol) and len(sol) == 6


def test_solve_univariate_rational():
    """Test solve_univariate for rational functions"""
    assert solve_univariate((x - y ** 3) / ((y ** 2) * sqrt(1 - y ** 2)), x) \
        == [y ** 3]


@XFAIL
def test_issue_7228():
    assert solve_univariate(
        4 ** (2 * (x ** 2) + 2 * x) - 8, x) == [-Rational(3, 2), S.Half]


@XFAIL
def test_issue_7190():
    assert solve_univariate(log(x - 3) + log(x + 3), x) == [sqrt(10)]


def test_solve_univariate_transcendental_1():
    assert solve_univariate(exp(x) - 3, x) == [log(3)]
    assert set(solve_univariate((a * x + b) * (exp(x) - 3), x)) \
        == set([-b / a, log(3)])
    assert solve_univariate(log(x) - 3, x) == [exp(3)]


@XFAIL
def test_solve_univariate_transcendental_2():
    assert set(solve_univariate(exp(x) + exp(-x) - y, x)) in [set([
        log(y / 2 - sqrt(y ** 2 - 4) / 2),
        log(y / 2 + sqrt(y ** 2 - 4) / 2),
    ]), set([
        log(y - sqrt(y ** 2 - 4)) - log(2),
        log(y + sqrt(y ** 2 - 4)) - log(2)]),
        set([
            log(y / 2 - sqrt((y - 2) * (y + 2)) / 2),
            log(y / 2 + sqrt((y - 2) * (y + 2)) / 2)])]
    assert solve_univariate(3 ** (x + 2), x) == []
    assert solve_univariate(3 ** (2 - x), x) == []
    assert solve_univariate(x + 2 ** x, x) == [-LambertW(log(2)) / log(2)]
    ans = solve_univariate(3 * x + 5 + 2 ** (-5 * x + 3), x)
    assert len(ans) == 1 and ans[0].expand() == \
        -Rational(5, 3) + LambertW(-10240 * 2 **
                                   (S(1) / 3) * log(2) / 3) / (5 * log(2))
    assert solve_univariate(5 * x - 1 + 3 * exp(2 - 7 * x), x) == \
        [Rational(1, 5) + LambertW(-21 * exp(Rational(3, 5)) / 5) / 7]
    assert solve_univariate(2 * x + 5 + log(3 * x - 2), x) == \
        [Rational(2, 3) + LambertW(2 * exp(-Rational(19, 3)) / 3) / 2]
    assert solve_univariate(3 * x + log(4 * x), x) == [LambertW(Rational(3, 4)) / 3]
    assert set(solve_univariate((2 * x + 8) * (8 + exp(x)), x)
               ) == set([S(-4), log(8) + pi * I])
    eq = 2 * exp(3 * x + 4) - 3
    ans = solve_univariate(eq, x)  # this generated a failure in flatten
    assert len(ans) == 3 and all(eq.subs(x, a).n(chop=True) == 0 for a in ans)
    assert solve_univariate(2 * log(3 * x + 4) - 3, x) == [(exp(Rational(3, 2)) - 4) / 3]
    assert solve_univariate(exp(x) + 1, x) == [pi * I]

    eq = 2 * (3 * x + 4) ** 5 - 6 * 7 ** (3 * x + 9)
    result = solve_univariate(eq, x)
    ans = [(log(2401) + 5 * LambertW(-
                                     log(7 ** (7 * 3 ** Rational(1, 5) / 5)))) / (3 * log(7)) / -1]
    assert result == ans
    # it works if expanded, too
    assert solve_univariate(eq.expand(), x) == result

    assert solve_univariate(z * cos(x) - y, x) == [-acos(y / z) + 2 * pi, acos(y / z)]
    assert solve_univariate(
        z * cos(2 * x) - y, x) == [-acos(y / z) / 2 + pi, acos(y / z) / 2]
    assert solve_univariate(z * cos(sin(x)) - y, x) == [
        asin(acos(y / z) - 2 * pi) + pi, -asin(acos(y / z)) + pi,
        -asin(acos(y / z) - 2 * pi), asin(acos(y / z))]

    assert solve_univariate(z * cos(x), x) == [pi / 2, 3 * pi / 2]

    # issue 4508
    assert solve_univariate(
        y - b * x / (a + x), x) in [[-a * y / (y - b)], [a * y / (b - y)]]
    assert solve_univariate(y - b * exp(a / x), x) == [a / log(y / b)]
    # issue 4507
    assert solve_univariate(
        y - b / (1 + a * x), x) in [[(b - y) / (a * y)], [-((y - b) / (a * y))]]
    # issue 4506
    assert solve_univariate(y - a * x ** b, x) == [(y / a) ** (1 / b)]
    # issue 4505
    assert solve_univariate(z ** x - y, x) == [log(y) / log(z)]
    # issue 4504
    assert solve_univariate(2 ** x - 10, x) == [log(10) / log(2)]
    # issue 6744
    assert solve_univariate(x * y) == [{x: 0}, {y: 0}]
    assert solve_univariate([x * y]) == [{x: 0}, {y: 0}]
    assert solve_univariate(x ** y - 1) == [{x: 1}, {y: 0}]
    assert solve_univariate([x ** y - 1]) == [{x: 1}, {y: 0}]
    assert solve_univariate(
        x * y * (x ** 2 - y ** 2)) == [{x: 0}, {x: -y}, {x: y}, {y: 0}]
    assert solve_univariate(
        [x * y * (x ** 2 - y ** 2)]) == [{x: 0}, {x: -y}, {x: y}, {y: 0}]
    # issue 4739
    assert solve_univariate(exp(log(5) * x) - 2 ** x, x) == [0]

    # misc
    # make sure that the right variables is picked up in tsolve_univariate
    raises(NotImplementedError, lambda: solve_univariate((exp(x) + 1) ** x))

    # shouldn't generate a GeneratorsNeeded error in _tsolve_univariate when the NaN is generated
    # for eq_down. Actual answers, as determined numerically are approx. +/-
    # 0.83
    assert solve_univariate(
        sinh(x) * sinh(sinh(x)) + cosh(x) * cosh(sinh(x)) - 3) is not None

    # watch out for recursive loop in tsolve_univariate
    raises(NotImplementedError, lambda: solve_univariate((x + 2) ** y * x - 3, x))

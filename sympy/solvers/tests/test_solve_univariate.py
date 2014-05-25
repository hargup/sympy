from sympy import (
    Abs, And, Derivative, Dummy, Eq, Float, Function, Gt, I, Integral,
    LambertW, Lt, Matrix, Or, Piecewise, Poly, Q, Rational, S, Symbol,
    Wild, acos, asin, atan, atanh, cos, cosh, diff, erf, erfinv, erfc,
    erfcinv, erf2, erf2inv, exp, expand, im, log, pi, re, sec, sin,
    sinh, solve, solve_linear, sqrt, sstr, symbols, sympify, tan, tanh,
    root, simplify, atan2, arg, Mul, SparseMatrix, ask)

from sympy.core.function import nfloat
from sympy.solvers import solve_linear_system, solve_linear_system_LU, \
    solve_undetermined_coeffs
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
    assert invert(x*3, x) == [x/3]

    assert invert(exp(x), x) == [log(x)]
    assert invert(exp(3*x), x) == [log(x)/3]
    assert invert(exp(x + 3), x) == [log(x) - 3]

    assert invert(log(x), x) == [exp(x)]
    assert invert(log(3*x), x) == [exp(x)/3]
    assert invert(log(x + 3), x) == [exp(x) - 3]


@XFAIL
def test_fail_invert():
    assert invert(x*log(x), x)


def test_polynomial():
    assert solve_univariate(3*x - 2, x) == [Rational(2, 3)]

    assert set(solve_univariate(x**2 - 1, x)) == set([-S(1), S(1)])

    assert solve_univariate(x - y**3, x) == [y**3]
    assert set(solve_univariate(x - y**3, y)) == set([
        (-x**Rational(1, 3))/2 + I*sqrt(3)*x**Rational(1, 3)/2,
        x**Rational(1, 3),
        (-x**Rational(1, 3))/2 - I*sqrt(3)*x**Rational(1, 3)/2,
    ])

    a11, a12, a21, a22, b1, b2 = symbols('a11,a12,a21,a22,b1,b2')

    assert set(solve_univariate(x**3 - 15*x - 4, x)) == set([
        -2 + 3**Rational(1, 2),
        S(4),
        -2 - 3**Rational(1, 2)
    ])

    assert set(solve_univariate((x**2 - 1)**2 - a, x)) == \
        set([sqrt(1 + sqrt(a)), -sqrt(1 + sqrt(a)),
             sqrt(1 - sqrt(a)), -sqrt(1 - sqrt(a))])


def test_no_sol():
    assert solve(4, x) == []


def test_solve_polynomial_cv_1a():
    """
    Test for solving on equations that can be converted to
    a polynomial equation using the change of variable y -> x**Rational(p, q)
    """
    assert solve_as_poly(sqrt(x) - 1, x) == [1]
    assert solve_as_poly(sqrt(x) - 2, x) == [4]
    assert solve_as_poly(x**Rational(1, 4) - 2, x) == [16]
    assert solve_as_poly(x**Rational(1, 3) - 3, x) == [27]


def test_solve_polynomial_multiple_gens():
    """
    Test for solving on equations that can be converted to a polynomial
    equation multiplying both sides of the equation by x**m
    """
    assert solve_as_poly(sqrt(x) + x**Rational(1, 3) +
                         x**Rational(1, 4), x) == [0]
    assert solve_as_poly(x + 1/x - 1, x) in \
        [[Rational(1, 2) + I*sqrt(3)/2, Rational(1, 2) - I*sqrt(3)/2],
         [Rational(1, 2) - I*sqrt(3)/2, Rational(1, 2) + I*sqrt(3)/2]]


@XFAIL
def test_solve_polnomoial_irration_deg():
    assert solve_as_poly(x**pi - 1, x)


def test_solve_polynomial_symbolic_param():
    assert set(solve_as_poly(4*x*(1 - a*sqrt(x)), x)) == set([S(0), 1/a**2])


def test_solve_polynomial_cv_1b():
    assert set(solve_as_poly(x * (x**(S(1)/3) - 3), x)) == set([S(0), S(27)])

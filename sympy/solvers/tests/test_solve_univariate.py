from sympy import (
    Abs, And, Derivative, Dummy, Eq, Float, Function, Gt, I, Integral,
    LambertW, Lt, Matrix, Or, Piecewise, Poly, Q, Rational, S, Symbol,
    Wild, acos, asin, atan, atanh, cos, cosh, diff, erf, erfinv, erfc,
    erfcinv, erf2, erf2inv, exp, expand, im, log, pi, re, sec, sin,
    sinh, sqrt, sstr, symbols, sympify, tan, tanh,
    simplify, atan2, arg, Mul, SparseMatrix, ask)

from sympy.core.function import nfloat
from sympy.solvers.solvers import _invert, unrad, checksol, posify, _ispow, \
    det_quick, det_perm, det_minor

from sympy.polys.rootoftools import RootOf

from sympy.utilities.pytest import slow, XFAIL, raises, skip
from sympy.utilities.randtest import test_numerically as tn

# from sympy.abc import a, b, x, y, z, q, m
a = Symbol('a', real=True)
b = Symbol('b', real=True)
x = Symbol('x', real=True)
y = Symbol('y', real=True)
z = Symbol('z', real=True)
q = Symbol('q', real=True)
m = Symbol('m', real=True)

from sympy.solvers.solve_univariate import solve_univariate_real, invert, \
    solve_as_poly, subexpression_checking, solve_univariate_complex

from sympy.solvers import solve


# TODO: fix the pep8 error in the solvers code and the test
# They are irritating and it is tempting to solve them along with writing the
# code


def test_invert():
    x = Symbol('x', real= True)
    assert invert(x + 3, x, y) == [y - 3]
    assert invert(x*3, x, y) == [y / 3]

    assert invert(exp(x), x, y) == [log(y)]
    assert invert(exp(3*x), x, y) == [log(y) / 3]
    assert invert(exp(x + 3), x, y) == [log(y) - 3]

    assert invert(exp(x) + 3, x, y) == [log(y - 3)]
    assert invert(exp(x)*3, x, y) == [log(y / 3)]

    assert invert(log(x), x, y) == [exp(y)]
    assert invert(log(3*x), x, y) == [exp(y) / 3]
    assert invert(log(x + 3), x, y) == [exp(y) - 3]

    assert invert(Abs(x), x, y) == [-y, y]

    assert invert(2**x, x, y) == [log(y)/log(2)]
    assert invert(2**exp(x), x, y) == [log(log(y)/log(2))]

    assert invert(x**2, x, y) == [sqrt(y), -sqrt(y)]
    assert invert(x**Rational(1, 2), x, y) == [y**2]

    raises(ValueError, lambda: invert(x**pi, x, y))

    x = Symbol('x', positive = True)
    assert invert(x**pi, x, y) == [y**(1/pi)]


@XFAIL
def test_invert_lambert():
    assert invert(x*exp(x), x) == LambertW(x)


def test_subexpression_checking():
    assert subexpression_checking(1/(1 + (1/(x+1))**2), x, -1) is False
    assert subexpression_checking(x**2, x, 0) is True


def test_polynomial():
    assert solve_univariate_real(3*x - 2, x) == [Rational(2, 3)]

    assert set(solve_univariate_real(x**2 - 1, x)) == set([-S(1), S(1)])
    assert solve_univariate_real(x - y**3, x) == [y ** 3]

    a11, a12, a21, a22, b1, b2 = symbols('a11,a12,a21,a22,b1,b2')

    assert set(solve_univariate_real(x**3 - 15*x - 4, x)) == set([
        -2 + 3 ** Rational(1, 2),
        S(4),
        -2 - 3 ** Rational(1, 2)])


def test_solve_rational():
    assert solve_univariate_real(1/x + 1, x) == [-S.One]
    assert solve_univariate_real(1/exp(x) - 1, x) == [0]


def test_solve_as_poly_gen_is_pow():
    assert solve_as_poly(sqrt(1) + 1, x) == []


def test_no_sol_simple():
    assert solve_univariate_real(4, x) == []
    assert solve_univariate_real(exp(x), x) == []


def test_no_sol_rational1():
    assert solve_univariate_real(1/x, x) == []
    assert solve_univariate_real(-(1 + x)/(2 + x)**2 + 1/(2 + x), x) == []


def test_no_sol_zero():
    assert solve_univariate_real(-x**2 - 2*x + (x + 1)**2 - 1, x) == []


def test_no_sol_rational_extragenous():
    # Simplification is messing up with the solutions of these equations. For example
    # for the one below a root exists for -1 but -1 is not in the domain of the function.
    assert solve_univariate_real((x/(x + 1) + 3)**(-2), x) == []
    assert solve_univariate_real((x - 1)/(1 + 1/(x - 1)), x) == []


def test_solve_polynomial_cv_1a():
    """
    Test for solving on equations that can be converted to
    a polynomial equation using the change of variable y -> x**Rational(p, q)
    """
    assert solve_as_poly(sqrt(x) - 1, x) == [1]
    assert solve_as_poly(sqrt(x) - 2, x) == [4]
    assert solve_as_poly(x**Rational(1, 4) - 2, x) == [16]
    assert solve_as_poly(x**Rational(1, 3) - 3, x) == [27]


@XFAIL
def test_solve_polynomial_multiple_gens():
    assert solve_as_poly(sqrt(x) + x**Rational(1, 3) +
                         x**Rational(1, 4), x) == [0]


@XFAIL
def test_solve_polnomoial_irration_deg():
    assert solve_univariate_real(x**pi - 1, x) == 1


@XFAIL
def test_solve_polynomial_symbolic_param():
    a = Symbol('a', positive=True)

    assert set(solve_univariate_real((x**2 - 1)**2 - a, x)) == \
        set([sqrt(1 + sqrt(a)), -sqrt(1 + sqrt(a)),
             sqrt(1 - sqrt(a)), -sqrt(1 - sqrt(a))])

    assert set(
        solve_as_poly(4 *x*(1 - a * sqrt(x)), x)) == set([S(0), 1 / a ** 2])


def test_solve_polynomial_cv_1b():
    assert set(
        solve_as_poly(x*(x**(S(1) / 3) - 3), x)) == set([S(0), S(27)])


def test_solve_univariate_real_rational():
    """Test solve_univariate_real for rational functions"""
    assert solve_univariate_real((x - y**3) / ((y**2)*sqrt(1 - y**2)), x) \
        == [y**3]


@XFAIL
def test_issue_7228():
    # In theory this test should have been passed, but the Poly function screws it up.
    # It creates two generators out of it but spliting 4**(a + b) to 4**a*4**b
    assert solve_univariate_real( 4**(2*(x**2) + 2*x) - 8, x) == \
            [-Rational(3, 2), S.Half]


@XFAIL
def test_issue_7190():
    assert solve_univariate_real(log(x - 3) + log(x + 3), x) == [sqrt(10)]


def test_solve_univariate_real_transcendental_1():
    assert solve_univariate_real(exp(x) - 3, x) == [log(3)]
    assert set(solve_univariate_real((a*x + b)*(exp(x) - 3), x)) \
        == set([-b / a, log(3)])
    assert solve_univariate_real(log(x) - 3, x) == [exp(3)]


def test_quintics_1():
    f = x**5 - 110*x**3 - 55*x**2 + 2310*x + 979
    s = solve_univariate_real(f, x)
    for root in s:
        res = f.subs(x, root.n()).n()
        assert tn(res, 0)


def test_solve_rational2():
    """Test solve for rational functions"""
    assert solve_univariate_real(( x - y**3)/((y**2)*sqrt(1 - y**2)), x) == [y**3]


def test_issue_4793_1():
    assert solve_univariate_real(x*(1 - 5/x), x) == [5]
    assert solve_univariate_real(-(1 + x)/(2 + x)**2 + 1/(2 + x), x) == []
    assert solve_univariate_real(-x**2 - 2*x + (x + 1)**2 - 1, x) == []
    assert solve_univariate_real((x/(x + 1) + 3)**(-2), x) == []
    assert solve_univariate_real(x/sqrt(x**2 + 1), x) == [0]

    y = Symbol('y', real=True)
    assert solve_univariate_real(exp(x) - y, x) == []

    y = Symbol('y', positive=True)
    assert solve_univariate_real(exp(x) - y, x) == [log(y)]


@XFAIL
def test_issue_4793_2():
    assert solve_univariate_real(x + sqrt(x) - 2, x) == [1]
    # I don't know underwhat general techniques this will fit. First I though of rewriting
    # x as sqrt(x)**2 but sqrt(x)**2 is multivalued with values x and -x so the equation
    # is basically this equation is a set of two equations.

    assert solve_univariate_real(log(x**2) - y**2/exp(x), x, y, set=True) == \
        ([y], set([
            (-sqrt(exp(x)*log(x**2)),),
            (sqrt(exp(x)*log(x**2)),)]))
    assert solve_univariate_real(x**2*z**2 - z**2*y**2) == [{x: -y}, {x: y}, {z: 0}]
    assert solve_univariate_real((x - 1)/(1 + 1/(x - 1))) == []
    assert solve_univariate_real(x**(y*z) - x, x) == [1]
    raises(NotImplementedError, lambda: solve_univariate_real(log(x) - exp(x), x))
    raises(NotImplementedError, lambda: solve_univariate_real(2**x - exp(x) - 3))


def test_atan2():
    assert solve_univariate_real(atan2(x, 2) - pi/3, x) == [2*sqrt(3)]


def test_errorinverses():
    assert solve_univariate_real(erf(x) - S.One/2, x)==[erfinv(S.One/2)]
    assert solve_univariate_real(erf(x) - S.One*2, x)==[]

    assert solve_univariate_real(erfinv(x) - 2, x)==[erf(2)]

    assert solve_univariate_real(erfc(x) - S.One, x)==[erfcinv(S.One)]
    assert solve_univariate_real(erfc(x) - S.One*3, x)==[]

    assert solve_univariate_real(erfcinv(x) - 2,x)==[erfc(2)]


def test_piecewise():
    assert set(solve_univariate_real(Piecewise((x - 2, Gt(x, 2)), (2 - x, True)) - 3, x)) == set([-1, 5])


def test_solve_univariate_complex_polynomial():
    from sympy.abc import x, a, b, c
    assert set(solve_univariate_complex(a*x**2 + b*x + c, x)) == \
            set([-b/(2*a) - sqrt(-4*a*c + b**2)/(2*a), -b/(2*a) + sqrt(-4*a*c + b**2)/(2*a)])


@XFAIL
def test_solve_univariate_complex_rational():
    from sympy.abc import x, a, b, c
    assert solve_univariate_complex((x - 1)*(x - I)/(x - 3), x) == [1, I]
    # The test is failing because roots return the solutions in a particularly
    # complex form. The answer given by it is
    # [1/2 + I/2 - sqrt(2)*sqrt(-I)/2, 1/2 + sqrt(2)*sqrt(-I)/2 + I/2]
    # either I have to simplify the output of the solve
    # for test for mathematical equivalence rather than structural equivalence


@XFAIL
def test_solve_univariate_complex_log():
    from sympy.abc import x
    eq = 4*3**(5*x + 2) - 7
    ans = solve_univariate_complex(eq, x)
    ans2 = solve(eq, x)
    assert len(ans) == 5 and all(eq.subs(x, a).n(chop=True) == 0 for a in ans)

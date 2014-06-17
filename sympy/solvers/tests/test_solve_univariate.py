from sympy import (
    Abs, And, Derivative, Dummy, Eq, Float, Function, Gt, I, Integral,
    LambertW, Lt, Matrix, Or, Piecewise, Poly, Q, Rational, S, Symbol, Wild,
    acos, asin, atan, atanh, cos, cosh, diff, erf, erfinv, erfc, erfcinv, erf2,
    erf2inv, exp, expand, im, log, pi, re, sec, sin, sinh, sqrt, sstr, symbols,
    sympify, tan, tanh, simplify, atan2, arg, Mul, SparseMatrix, ask, tan,
    Lambda, imageset, cot, acot)

from sympy.core.function import nfloat
from sympy.solvers.solvers import _invert, unrad, checksol, posify, _ispow, \
    det_quick, det_perm, det_minor

from sympy.polys.rootoftools import RootOf

from sympy.sets import FiniteSet

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
    solve_as_poly, domain_check, solve_univariate_complex

from sympy.solvers import solve


# TODO: fix the pep8 error in the solvers code and the test
# They are irritating and it is tempting to solve them along with writing the
# code


def test_invert():
    x = Symbol('x', real= True)
    assert invert(x + 3, x, y) == FiniteSet(y - 3)
    assert invert(x*3, x, y) == FiniteSet(y / 3)

    assert invert(exp(x), x, y) == FiniteSet(log(y))
    assert invert(exp(3*x), x, y) == FiniteSet(log(y) / 3)
    assert invert(exp(x + 3), x, y) == FiniteSet(log(y) - 3)

    assert invert(exp(x) + 3, x, y) == FiniteSet(log(y - 3))
    assert invert(exp(x)*3, x, y) == FiniteSet(log(y / 3))

    assert invert(log(x), x, y) == FiniteSet(exp(y))
    assert invert(log(3*x), x, y) == FiniteSet(exp(y) / 3)
    assert invert(log(x + 3), x, y) == FiniteSet(exp(y) - 3)

    assert invert(Abs(x), x, y) == FiniteSet(-y, y)

    assert invert(2**x, x, y) == FiniteSet(log(y)/log(2))
    assert invert(2**exp(x), x, y) == FiniteSet(log(log(y)/log(2)))

    assert invert(x**2, x, y) == FiniteSet(sqrt(y), -sqrt(y))
    assert invert(x**Rational(1, 2), x, y) == FiniteSet(y**2)

    raises(ValueError, lambda: invert(x**pi, x, y))

    x = Symbol('x', positive = True)
    assert invert(x**pi, x, y) == FiniteSet(y**(1/pi))


def test_invert_tan_cot():
    from sympy.abc import x, y, n
    raises(NotImplementedError, lambda: invert(tan(cot(x)), x))
    raises(NotImplementedError, lambda: invert(tan(sin(x)), x))
    raises(NotImplementedError, lambda: invert(cot(cot(x)), x))

    assert invert(tan(x), x, y) == \
            imageset(Lambda(n, n*pi + atan(y)), S.Integers)
    assert invert(tan(exp(x)), x, y) == \
                  imageset(Lambda(n, log(n*pi + atan(y))), S.Integers)

    assert invert(cot(x), x, y) == \
            imageset(Lambda(n, n*pi + acot(y)), S.Integers)
    assert invert(cot(exp(x)), x, y) == \
                  imageset(Lambda(n, log(n*pi + acot(y))), S.Integers)


def test_invert_sin_cos():
    from sympy.abc import x, y, n
    raises(NotImplementedError, lambda: invert(sin(sin(x)), x))
    raises(NotImplementedError, lambda: invert(sin(tan(x)), x))

    assert invert(sin(x), x, y) == \
            imageset(Lambda(n, n*pi + asin(y)*(-S.One)**(n)), S.Integers)
    assert invert(cos(x), x, y) == \
            imageset(Lambda(n, n*pi + pi/2 + ((-S.One)**(n))*(acos(y) - pi/2)), S.Integers)

    assert invert(sin(exp(x)), x, y) == \
                  imageset(Lambda(n, log(n*pi + asin(y)*(-S.One)**(n))), S.Integers)


@XFAIL
def test_invert_lambert():
    assert invert(x*exp(x), x) == LambertW(x)


def test_domain_check():
    assert domain_check(1/(1 + (1/(x+1))**2), x, -1) is False
    assert domain_check(x**2, x, 0) is True


def test_polynomial():
    assert solve_univariate_real(3*x - 2, x) == FiniteSet(Rational(2, 3))

    assert solve_univariate_real(x**2 - 1, x) == FiniteSet(-S(1), S(1))
    assert solve_univariate_real(x - y**3, x) == FiniteSet(y ** 3)

    a11, a12, a21, a22, b1, b2 = symbols('a11,a12,a21,a22,b1,b2')

    assert set(solve_univariate_real(x**3 - 15*x - 4, x)) == set(FiniteSet(
        -2 + 3 ** Rational(1, 2),
        S(4),
        -2 - 3 ** Rational(1, 2)))


def test_solve_rational():
    assert solve_univariate_real(1/x + 1, x) == FiniteSet(-S.One)
    assert solve_univariate_real(1/exp(x) - 1, x) == FiniteSet(0)


def test_solve_as_poly_gen_is_pow():
    assert solve_as_poly(sqrt(1) + 1, x) == FiniteSet()


def test_no_sol_simple():
    assert solve_univariate_real(4, x) == FiniteSet()
    assert solve_univariate_real(exp(x), x) == FiniteSet()


def test_no_sol_rational1():
    assert solve_univariate_real(1/x, x) == FiniteSet()
    assert solve_univariate_real(-(1 + x)/(2 + x)**2 + 1/(2 + x), x) == FiniteSet()


def test_no_sol_zero():
    assert solve_univariate_real(-x**2 - 2*x + (x + 1)**2 - 1, x) == FiniteSet()


def test_no_sol_rational_extragenous():
    # Simplification is messing up with the solutions of these equations. For example
    # for the one below a root exists for -1 but -1 is not in the domain of the function.
    assert solve_univariate_real((x/(x + 1) + 3)**(-2), x) == FiniteSet()
    assert solve_univariate_real((x - 1)/(1 + 1/(x - 1)), x) == FiniteSet()


def test_solve_polynomial_cv_1a():
    """
    Test for solving on equations that can be converted to
    a polynomial equation using the change of variable y -> x**Rational(p, q)
    """
    assert solve_as_poly(sqrt(x) - 1, x) == FiniteSet(1)
    assert solve_as_poly(sqrt(x) - 2, x) == FiniteSet(4)
    assert solve_as_poly(x**Rational(1, 4) - 2, x) == FiniteSet(16)
    assert solve_as_poly(x**Rational(1, 3) - 3, x) == FiniteSet(27)


@XFAIL
def test_solve_polynomial_multiple_gens():
    assert solve_as_poly(sqrt(x) + x**Rational(1, 3) +
                         x**Rational(1, 4), x) == FiniteSet(0)


@XFAIL
def test_solve_polnomoial_irration_deg():
    assert solve_univariate_real(x**pi - 1, x) == 1


@XFAIL
def test_solve_polynomial_symbolic_param():
    a = Symbol('a', positive=True)

    assert set(solve_univariate_real((x**2 - 1)**2 - a, x)) == \
        set(FiniteSet(sqrt(1 + sqrt(a)), -sqrt(1 + sqrt(a)),
             sqrt(1 - sqrt(a)), -sqrt(1 - sqrt(a))))

    assert set(
        solve_as_poly(4 *x*(1 - a * sqrt(x)), x)) == set(FiniteSet(S(0), 1 / a ** 2))


def test_solve_polynomial_cv_1b():
    assert set(
        solve_as_poly(x*(x**(S(1) / 3) - 3), x)) == set(FiniteSet(S(0), S(27)))


def test_solve_univariate_real_rational():
    """Test solve_univariate_real for rational functions"""
    assert solve_univariate_real((x - y**3) / ((y**2)*sqrt(1 - y**2)), x) \
        == FiniteSet(y**3)


@XFAIL
def test_issue_7228():
    # In theory this test should have been passed, but the Poly function screws it up.
    # It creates two generators out of it but spliting 4**(a + b) to 4**a*4**b
    assert solve_univariate_real( 4**(2*(x**2) + 2*x) - 8, x) == \
            FiniteSet(-Rational(3, 2), S.Half)


@XFAIL
def test_issue_7190():
    assert solve_univariate_real(log(x - 3) + log(x + 3), x) == FiniteSet(sqrt(10))


def test_solve_univariate_real_transcendental_1():
    assert solve_univariate_real(exp(x) - 3, x) == FiniteSet(log(3))
    assert set(solve_univariate_real((a*x + b)*(exp(x) - 3), x)) \
        == set(FiniteSet(-b / a, log(3)))
    assert solve_univariate_real(log(x) - 3, x) == FiniteSet(exp(3))


def test_quintics_1():
    f = x**5 - 110*x**3 - 55*x**2 + 2310*x + 979
    s = solve_univariate_real(f, x)
    for root in s:
        res = f.subs(x, root.n()).n()
        assert tn(res, 0)


def test_solve_rational2():
    """Test solve for rational functions"""
    assert solve_univariate_real(( x - y**3)/((y**2)*sqrt(1 - y**2)), x) == FiniteSet(y**3)


def test_issue_4793_1():
    assert solve_univariate_real(x*(1 - 5/x), x) == FiniteSet(5)
    assert solve_univariate_real(-(1 + x)/(2 + x)**2 + 1/(2 + x), x) == FiniteSet()
    assert solve_univariate_real(-x**2 - 2*x + (x + 1)**2 - 1, x) == FiniteSet()
    assert solve_univariate_real((x/(x + 1) + 3)**(-2), x) == FiniteSet()
    assert solve_univariate_real(x/sqrt(x**2 + 1), x) == FiniteSet(0)

    y = Symbol('y', real=True)
    assert solve_univariate_real(exp(x) - y, x) == FiniteSet()

    y = Symbol('y', positive=True)
    assert solve_univariate_real(exp(x) - y, x) == FiniteSet(log(y))


@XFAIL
def test_issue_4793_2():
    assert solve_univariate_real(x + sqrt(x) - 2, x) == FiniteSet(1)
    # I don't know underwhat general techniques this will fit. First I though of rewriting
    # x as sqrt(x)**2 but sqrt(x)**2 is multivalued with values x and -x so the equation
    # is basically this equation is a set of two equations.

    assert solve_univariate_real(log(x**2) - y**2/exp(x), x, y, set=True) == \
        (FiniteSet(y), set(FiniteSet(
            (-sqrt(exp(x)*log(x**2)),),
            (sqrt(exp(x)*log(x**2)),))))
    assert solve_univariate_real(x**2*z**2 - z**2*y**2) == FiniteSet({x: -y}, {x: y}, {z: 0})
    assert solve_univariate_real((x - 1)/(1 + 1/(x - 1))) == FiniteSet()
    assert solve_univariate_real(x**(y*z) - x, x) == FiniteSet(1)
    raises(NotImplementedError, lambda: solve_univariate_real(log(x) - exp(x), x))
    raises(NotImplementedError, lambda: solve_univariate_real(2**x - exp(x) - 3))


def test_atan2():
    assert solve_univariate_real(atan2(x, 2) - pi/3, x) == FiniteSet(2*sqrt(3))


def test_errorinverses():
    assert solve_univariate_real(erf(x) - S.One/2, x) == FiniteSet(erfinv(S.One/2))
    assert solve_univariate_real(erf(x) - S.One*2, x) == FiniteSet()

    assert solve_univariate_real(erfinv(x) - 2, x) == FiniteSet(erf(2))

    assert solve_univariate_real(erfc(x) - S.One, x) == FiniteSet(erfcinv(S.One))
    assert solve_univariate_real(erfc(x) - S.One*3, x) == FiniteSet()

    assert solve_univariate_real(erfcinv(x) - 2,x) == FiniteSet(erfc(2))


def test_piecewise():
    assert set(solve_univariate_real(Piecewise((x - 2, Gt(x, 2)), (2 - x, True)) - 3, x)) == set(FiniteSet(-1, 5))


def test_solve_univariate_complex_polynomial():
    from sympy.abc import x, a, b, c
    assert set(solve_univariate_complex(a*x**2 + b*x + c, x)) == \
            set(FiniteSet(-b/(2*a) - sqrt(-4*a*c + b**2)/(2*a), -b/(2*a) + sqrt(-4*a*c + b**2)/(2*a)))


@XFAIL
def test_solve_univariate_complex_rational():
    from sympy.abc import x, a, b, c
    assert solve_univariate_complex((x - 1)*(x - I)/(x - 3), x) == FiniteSet(1, I)
    # The test is failing because roots return the solutions in a particularly
    # complex form. The answer given by it is
    # FiniteSet(1/2 + I/2 - sqrt(2)*sqrt(-I)/2, 1/2 + sqrt(2)*sqrt(-I)/2 + I/2)
    # either I have to simplify the output of the solve
    # for test for mathematical equivalence rather than structural equivalence


@XFAIL
def test_solve_univariate_complex_log():
    from sympy.abc import x
    eq = 4*3**(5*x + 2) - 7
    ans = solve_univariate_complex(eq, x)
    ans2 = solve(eq, x)
    assert len(ans) == 5 and all(eq.subs(x, a).n(chop=True) == 0 for a in ans)


def test_solve_trig():
    from sympy.abc import n
    assert solve_univariate_real(sin(x), x) == imageset(Lambda(n, n*pi), S.Integers)
    assert solve_univariate_real(sin(x) - 1, x) == imageset(Lambda(n, n*pi + (-1)**n*pi/2), S.Integers)
    # TODO: checkout if there can be general method to simplify n + ((-1)**n)/2 to 2*n + 1/2

    assert solve_univariate_real(cos(x), x) == imageset(Lambda(n, n*pi + pi/2), S.Integers)

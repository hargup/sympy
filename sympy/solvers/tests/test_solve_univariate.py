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
    solve_as_poly, solve_as_poly_gen_is_pow, subexpression_checking


# TODO: fix the pep8 error in the solvers code and the test
# They are irritating and it is tempting to solve them along with writing the
# code

def NS(e, n=15, **options):
    return sstr(sympify(e).evalf(n, **options), full_prec=True)

def cannot_solve(f, symbol):
    try:
        solve_univariate_real(f, symbol)
        return False
    except (ValueError):
        return True


def test_invert():
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
        -2 - 3 ** Rational(1, 2)
    ])

    # assert set(solve_univariate_real((x**2 - 1)**2 - a, x)) == \
    #     set([sqrt(1 + sqrt(a)), -sqrt(1 + sqrt(a)),
    #          sqrt(1 - sqrt(a)), -sqrt(1 - sqrt(a))])


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
    """
    Test for solving on equations that can be converted to a polynomial
    equation multiplying both sides of the equation by x**m
    """
    assert solve_as_poly(sqrt(x) + x**Rational(1, 3) +
                         x**Rational(1, 4), x) == [0]
    assert solve_as_poly(x + 1/x - 1, x) in \
        [[Rational(1, 2) + I*sqrt(3) / 2, Rational(1, 2) - I * sqrt(3) / 2],
         [Rational(1, 2) - I*sqrt(3) / 2, Rational(1, 2) + I * sqrt(3) / 2]]


def test_solve_polnomoial_irration_deg():
    assert cannot_solve(x**pi - 1, x)


@XFAIL
def test_solve_polynomial_symbolic_param():
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
    assert solve_univariate_real(
        4**(2*(x**2) + 2*x) - 8, x) == [-Rational(3, 2), S.Half]


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


@XFAIL
def test_solve_univariate_real_transcendental_2():
    assert set(solve_univariate_real(exp(x) + exp(-x) - y, x)) in [set([
        log(y / 2 - sqrt(y**2 - 4) / 2),
        log(y / 2 + sqrt(y**2 - 4) / 2),
    ]), set([
        log(y - sqrt(y**2 - 4)) - log(2),
        log(y + sqrt(y**2 - 4)) - log(2)]),
        set([
            log(y / 2 - sqrt((y - 2) * (y + 2)) / 2),
            log(y / 2 + sqrt((y - 2) * (y + 2)) / 2)])]
    assert solve_univariate_real(3 ** (x + 2), x) == []
    assert solve_univariate_real(3 ** (2 - x), x) == []
    assert solve_univariate_real(x + 2 ** x, x) == [-LambertW(log(2)) / log(2)]
    ans = solve_univariate_real(3 * x + 5 + 2 ** (-5 * x + 3), x)
    assert len(ans) == 1 and ans[0].expand() == \
        -Rational(5, 3) + LambertW(-10240 * 2 **
                                   (S(1) / 3) * log(2) / 3) / (5 * log(2))
    assert solve_univariate_real(5 * x - 1 + 3 * exp(2 - 7 * x), x) == \
        [Rational(1, 5) + LambertW(-21 * exp(Rational(3, 5)) / 5) / 7]
    assert solve_univariate_real(2 * x + 5 + log(3 * x - 2), x) == \
        [Rational(2, 3) + LambertW(2 * exp(-Rational(19, 3)) / 3) / 2]
    assert solve_univariate_real(3 * x + log(4 * x), x) == [LambertW(Rational(3, 4)) / 3]
    assert set(solve_univariate_real((2 * x + 8) * (8 + exp(x)), x)
               ) == set([S(-4), log(8) + pi * I])
    eq = 2 * exp(3 * x + 4) - 3
    ans = solve_univariate_real(eq, x)  # this generated a failure in flatten
    assert len(ans) == 3 and all(eq.subs(x, a).n(chop=True) == 0 for a in ans)
    assert solve_univariate_real(2 * log(3 * x + 4) - 3, x) == [(exp(Rational(3, 2)) - 4) / 3]
    assert solve_univariate_real(exp(x) + 1, x) == [pi * I]

    eq = 2 * (3 * x + 4) ** 5 - 6 * 7 ** (3 * x + 9)
    result = solve_univariate_real(eq, x)
    ans = [(log(2401) + 5 * LambertW(-
                                     log(7 ** (7 * 3 ** Rational(1, 5) / 5)))) / (3 * log(7)) / -1]
    assert result == ans
    # it works if expanded, too
    assert solve_univariate_real(eq.expand(), x) == result

    assert solve_univariate_real(z * cos(x) - y, x) == [-acos(y / z) + 2 * pi, acos(y / z)]
    assert solve_univariate_real(
        z * cos(2 * x) - y, x) == [-acos(y / z) / 2 + pi, acos(y / z) / 2]
    assert solve_univariate_real(z * cos(sin(x)) - y, x) == [
        asin(acos(y / z) - 2 * pi) + pi, -asin(acos(y / z)) + pi,
        -asin(acos(y / z) - 2 * pi), asin(acos(y / z))]

    assert solve_univariate_real(z * cos(x), x) == [pi / 2, 3 * pi / 2]

    # issue 4508
    assert solve_univariate_real(
        y - b * x / (a + x), x) in [[-a * y / (y - b)], [a * y / (b - y)]]
    assert solve_univariate_real(y - b * exp(a / x), x) == [a / log(y / b)]
    # issue 4507
    assert solve_univariate_real(
        y - b / (1 + a * x), x) in [[(b - y) / (a * y)], [-((y - b) / (a * y))]]
    # issue 4506
    assert solve_univariate_real(y - a * x ** b, x) == [(y / a) ** (1 / b)]
    # issue 4505
    assert solve_univariate_real(z ** x - y, x) == [log(y) / log(z)]
    # issue 4504
    assert solve_univariate_real(2 ** x - 10, x) == [log(10) / log(2)]
    # issue 6744
    assert solve_univariate_real(x * y) == [{x: 0}, {y: 0}]
    assert solve_univariate_real([x * y]) == [{x: 0}, {y: 0}]
    assert solve_univariate_real(x ** y - 1) == [{x: 1}, {y: 0}]
    assert solve_univariate_real([x ** y - 1]) == [{x: 1}, {y: 0}]
    assert solve_univariate_real(
        x * y * (x ** 2 - y ** 2)) == [{x: 0}, {x: -y}, {x: y}, {y: 0}]
    assert solve_univariate_real(
        [x * y * (x ** 2 - y ** 2)]) == [{x: 0}, {x: -y}, {x: y}, {y: 0}]
    # issue 4739
    assert solve_univariate_real(exp(log(5) * x) - 2 ** x, x) == [0]

    # misc
    # make sure that the right variables is picked up in tsolve_univariate_real
    raises(NotImplementedError, lambda: solve_univariate_real((exp(x) + 1) ** x))

    # shouldn't generate a GeneratorsNeeded error in _tsolve_univariate_real when the NaN is generated
    # for eq_down. Actual answers, as determined numerically are approx. +/-
    # 0.83
    assert solve_univariate_real(
        sinh(x) * sinh(sinh(x)) + cosh(x) * cosh(sinh(x)) - 3) is not None

    # watch out for recursive loop in tsolve_univariate_real
    raises(NotImplementedError, lambda: solve_univariate_real((x + 2) ** y * x - 3, x))


@XFAIL
def test_issue_4793():
    assert solve_univariate_real(x*(1 - 5/x), x) == [5]
    assert solve_univariate_real(x + sqrt(x) - 2, x) == [1]
    assert solve_univariate_real(-(1 + x)/(2 + x)**2 + 1/(2 + x), x) == []
    assert solve_univariate_real(-x**2 - 2*x + (x + 1)**2 - 1, x) == []
    assert solve_univariate_real((x/(x + 1) + 3)**(-2), x) == []
    assert solve_univariate_real(x/sqrt(x**2 + 1), x) == [0]
    assert solve_univariate_real(exp(x) - y, x) == [log(y)]
    assert solve_univariate_real(x**2 + x + sin(y)**2 + cos(y)**2 - 1, x) in [[0, -1], [-1, 0]]
    eq = 4*3**(5*x + 2) - 7
    ans = solve_univariate_real(eq, x)
    assert len(ans) == 5 and all(eq.subs(x, a).n(chop=True) == 0 for a in ans)
    assert solve_univariate_real(log(x**2) - y**2/exp(x), x, y, set=True) == \
        ([y], set([
            (-sqrt(exp(x)*log(x**2)),),
            (sqrt(exp(x)*log(x**2)),)]))
    assert solve_univariate_real(x**2*z**2 - z**2*y**2) == [{x: -y}, {x: y}, {z: 0}]
    assert solve_univariate_real((x - 1)/(1 + 1/(x - 1))) == []
    assert solve_univariate_real(x**(y*z) - x, x) == [1]
    raises(NotImplementedError, lambda: solve_univariate_real(log(x) - exp(x), x))
    raises(NotImplementedError, lambda: solve_univariate_real(2**x - exp(x) - 3))


@XFAIL
def test_PR1964():
    # issue 5171
    assert solve_univariate_real(sqrt(x), x) == solve_univariate_real(sqrt(x**3)) == [0]
    assert solve_univariate_real(sqrt(x - 1), x) == [1]
    # issue 4462
    a = Symbol('a')
    assert solve_univariate_real(-3*a/sqrt(x), x) == []
    # issue 4486
    assert solve_univariate_real(2*x/(x + 2) - 1, x) == [2]
    # issue 4496
    assert set(solve_univariate_real((x**2/(7 - x)).diff(x), x)) == set([S(0), S(14)])

    # issue 4497
    assert solve_univariate_real(1/(5 + x)**(S(1)/5) - 9, x) == [-295244/S(59049)]

    assert solve_univariate_real(sqrt(x) + sqrt(sqrt(x)) - 4, x) == [-9*sqrt(17)/2 + 49*S.Half]

    assert set(solve_univariate_real(sqrt(exp(x)) + sqrt(exp(-x)) - 4, x)) in \
        [
            set([2*log(-sqrt(3) + 2), 2*log(sqrt(3) + 2)]),
            set([log(-4*sqrt(3) + 7), log(4*sqrt(3) + 7)]),
        ]
    assert set(solve_univariate_real(exp(x) + exp(-x) - 4, x)) == \
        set([log(-sqrt(3) + 2), log(sqrt(3) + 2)])
    assert set(solve_univariate_real(x**y + x**(2*y) - 1, x)) == \
        set([(-S.Half + sqrt(5)/2)**(1/y), (-S.Half - sqrt(5)/2)**(1/y)])

    assert solve_univariate_real(exp(x/y)*exp(-z/y) - 2, y) == [(x - z)/log(2)]
    assert solve_univariate_real(
        x**z*y**z - 2, z) in [[log(2)/(log(x) + log(y))], [log(2)/(log(x*y))]]
    # if you do inversion too soon then multiple roots as for the following will
    # be missed, e.g. if exp(3*x) = exp(3) -> 3*x = 3
    E = S.Exp1
    assert set(solve_univariate_real(exp(3*x) - exp(3), x)) in [
        set([S(1), log(-E/2 - sqrt(3)*E*I/2), log(-E/2 + sqrt(3)*E*I/2)]),
        set([S(1), log(E*(-S(1)/2 - sqrt(3)*I/2)), log(E*(-S(1)/2 + sqrt(3)*I/2))]),
    ]

    # coverage test
    p = Symbol('p', positive=True)
    assert solve_univariate_real((1/p + 1)**(p + 1)) == []

    assert set(solve_univariate_real(x*(x - y/x), x)) == set([sqrt(y), -sqrt(y)])


@XFAIL
def test_issue_4671_4463_4467():
    assert solve_univariate_real((sqrt(x**2 - 1) - 2), x) in ([sqrt(5), -sqrt(5)],
                                           [-sqrt(5), sqrt(5)])
    assert set(solve_univariate_real((2**exp(y**2/x) + 2)/(x**2 + 15), y)) == set([
        -sqrt(x)*sqrt(-log(log(2)) + log(log(2) + I*pi)),
        sqrt(x)*sqrt(-log(log(2)) + log(log(2) + I*pi))])

    a = Symbol('a')
    E = S.Exp1
    assert solve_univariate_real(1 - log(a + 4*x**2), x) in (
        [-sqrt(-a + E)/2, sqrt(-a + E)/2],
        [sqrt(-a + E)/2, -sqrt(-a + E)/2]
    )
    assert solve_univariate_real(log(a**(-3) - x**2)/a, x) in (
        [-sqrt(-1 + a**(-3)), sqrt(-1 + a**(-3))],
        [sqrt(-1 + a**(-3)), -sqrt(-1 + a**(-3))],)
    assert solve_univariate_real(1 - log(a + 4*x**2), x) in (
        [-sqrt(-a + E)/2, sqrt(-a + E)/2],
        [sqrt(-a + E)/2, -sqrt(-a + E)/2],)
    assert set(solve_univariate_real((
        a**2 + 1) * (sin(a*x) + cos(a*x)), x)) == set([-pi/(4*a), 3*pi/(4*a)])
    assert solve_univariate_real(3 - (sinh(a*x) + cosh(a*x)), x) == [log(3)/a]
    assert set(solve_univariate_real(3 - (sinh(a*x) + cosh(a*x)**2), x)) == \
        set([log(-2 + sqrt(5))/a, log(-sqrt(2) + 1)/a,
        log(-sqrt(5) - 2)/a, log(1 + sqrt(2))/a])
    assert solve_univariate_real(atan(x) - 1, x) == [tan(1)]


@XFAIL
def test_unrad():
    # http://tutorial.math.lamar.edu/
    #        Classes/Alg/solve_univariate_realRadicalEqns.aspx#solve_univariate_real_Rad_Ex2_a
    assert solve_univariate_real(x - sqrt(x + 6), x) == [3]
    assert solve_univariate_real(x + sqrt(x - 4) - 4, x) == [4]
    assert solve_univariate_real(Eq(1, x + sqrt(2*x - 3))) == []
    assert set(solve_univariate_real(Eq(sqrt(5*x + 6) - 2, x))) == set([-S(1), S(2)])
    assert set(solve_univariate_real(Eq(sqrt(2*x - 1) - sqrt(x - 4), 2))) == set([S(5), S(13)])
    assert solve_univariate_real(Eq(sqrt(x + 7) + 2, sqrt(3 - x))) == [-6]
    # http://www.purplemath.com/modules/solve_univariate_realrad.htm
    assert solve_univariate_real((2*x - 5)**Rational(1, 3) - 3) == [16]
    assert solve_univariate_real((x**3 - 3*x**2)**Rational(1, 3) + 1 - x) == []
    assert set(solve_univariate_real(x + 1 - (x**4 + 4*x**3 - x)**Rational(1, 4))) == \
        set([-S(1)/2, -S(1)/3])
    assert set(solve_univariate_real(sqrt(2*x**2 - 7) - (3 - x))) == set([-S(8), S(2)])
    assert solve_univariate_real(sqrt(2*x + 9) - sqrt(x + 1) - sqrt(x + 4)) == [0]
    assert solve_univariate_real(sqrt(x + 4) + sqrt(2*x - 1) - 3*sqrt(x - 1)) == [5]
    assert solve_univariate_real(sqrt(x)*sqrt(x - 7) - 12) == [16]
    assert solve_univariate_real(sqrt(x - 3) + sqrt(x) - 3) == [4]
    assert solve_univariate_real(sqrt(9*x**2 + 4) - (3*x + 2)) == [0]
    assert solve_univariate_real(sqrt(x) - 2 - 5) == [49]
    assert solve_univariate_real(sqrt(x - 3) - sqrt(x) - 3) == []
    assert solve_univariate_real(sqrt(x - 1) - x + 7) == [10]
    assert solve_univariate_real(sqrt(x - 2) - 5) == [27]
    assert solve_univariate_real(sqrt(17*x - sqrt(x**2 - 5)) - 7) == [3]
    assert solve_univariate_real(sqrt(x) - sqrt(x - 1) + sqrt(sqrt(x))) == []

    # don't posify the expression in unrad and use _mexpand
    z = sqrt(2*x + 1)/sqrt(x) - sqrt(2 + 1/x)
    p = posify(z)[0]
    assert solve_univariate_real(p) == []
    assert solve_univariate_real(z) == []
    assert solve_univariate_real(z + 6*I) == [-S(1)/11]
    assert solve_univariate_real(p + 6*I) == []

    eq = sqrt(2 + I) + 2*I
    assert unrad(eq - x, x, all=True) == (x**4 + 4*x**2 + 8*x + 37, [], [])
    ans = (81*x**8 - 2268*x**6 - 4536*x**5 + 22644*x**4 + 63216*x**3 -
        31608*x**2 - 189648*x + 141358, [], [])
    r = sqrt(sqrt(2)/3 + 7)
    eq = sqrt(r) + r - x
    assert unrad(eq, all=1)
    r2 = sqrt(sqrt(2) + 21)/sqrt(3)
    assert r != r2 and r.equals(r2)
    assert unrad(eq - r + r2, all=True) == ans


@XFAIL
def test_unrad_slow():
    ans = solve_univariate_real(sqrt(x) + sqrt(x + 1) -
                sqrt(1 - x) - sqrt(2 + x), x)
    assert len(ans) == 1 and NS(ans[0])[:4] == '0.73'
    # the fence optimization problem
    # https://github.com/sympy/sympy/issues/4793#issuecomment-36994519
    F = Symbol('F')
    eq = F - (2*x + 2*y + sqrt(x**2 + y**2))
    X = solve_univariate_real(eq, x)[0]
    Y = solve_univariate_real((x*y).subs(x, X).diff(y), y)
    ans = 2*F/7 - sqrt(2)*F/14
    assert any((a - ans).expand().is_zero for a in Y)

    eq = S('''
        -x + (1/2 - sqrt(3)*I/2)*(3*x**3/2 - x*(3*x**2 - 34)/2 + sqrt((-3*x**3
        + x*(3*x**2 - 34) + 90)**2/4 - 39304/27) - 45)**(1/3) + 34/(3*(1/2 -
        sqrt(3)*I/2)*(3*x**3/2 - x*(3*x**2 - 34)/2 + sqrt((-3*x**3 + x*(3*x**2
        - 34) + 90)**2/4 - 39304/27) - 45)**(1/3))''')
    raises(NotImplementedError, lambda: solve_univariate_real(eq)) # not other code errors


@XFAIL
def test_issue_4463():
    assert solve_univariate_real(-a*x + 2*x*log(x), x) == [exp(a/2)]
    assert solve_univariate_real(a/x + exp(x/2), x) == [2*LambertW(-a/2)]
    assert solve_univariate_real(x**x) == []
    assert solve_univariate_real(x**x - 2) == [exp(LambertW(log(2)))]
    assert solve_univariate_real(((x - 3)*(x - 2))**((x - 3)*(x - 4))) == [2]
    assert solve_univariate_real(
        (a/x + exp(x/2)).diff(x), x) == [4*LambertW(sqrt(2)*sqrt(a)/4)]


@XFAIL
def test_issue_6056():
    assert solve_univariate_real(tanh(x + 3)*tanh(x - 3) - 1, x) == []
    assert set([simplify(w) for w in solve_univariate_real(tanh(x - 1)*tanh(x + 1) + 1), x]) == set([
        -log(2)/2 + log(1 - I),
        -log(2)/2 + log(-1 - I),
        -log(2)/2 + log(1 + I),
        -log(2)/2 + log(-1 + I),])
    assert set([simplify(w) for w in solve_univariate_real((tanh(x + 3)*tanh(x - 3) + 1)**2, x)]) == set([
        -log(2)/2 + log(1 - I),
        -log(2)/2 + log(-1 - I),
        -log(2)/2 + log(1 + I),
        -log(2)/2 + log(-1 + I),])


@XFAIL
def test_issue_6060():
    x = Symbol('x')
    absxm3 = Piecewise(
        (x - 3, S(0) <= x - 3),
        (3 - x, S(0) > x - 3)
    )
    y = Symbol('y')
    assert solve_univariate_real(absxm3 - y, x) == [
        Piecewise((-y + 3, S(0) > -y), (S.NaN, True)),
        Piecewise((y + 3, S(0) <= y), (S.NaN, True))
    ]


    y = Symbol('y', positive=True)
    assert solve_univariate_real(absxm3 - y, x) == [-y + 3, y + 3]


@XFAIL
def test_issue_6605():
    x = symbols('x')
    assert solve_univariate_real(4**(x/2) - 2**(x/3), x) == [0]
    # while the first one passed, this one failed
    x = symbols('x', real=True)
    assert solve_univariate_real(5**(x/2) - 2**(x/3), x) == [0]
    b = sqrt(6)*sqrt(log(2))/sqrt(log(5))
    assert solve_univariate_real(5**(x/2) - 2**(3/x), x) == [-b, b]


@XFAIL
def test_issue_6644():
    eq = -sqrt((m - q)**2 + (-m/(2*q) + S(1)/2)**2) + sqrt((-m**2/2 - sqrt(
    4*m**4 - 4*m**2 + 8*m + 1)/4 - S(1)/4)**2 + (m**2/2 - m - sqrt(
    4*m**4 - 4*m**2 + 8*m + 1)/4 - S(1)/4)**2)
    assert solve_univariate_real(eq, q) == [
        m**2/2 - sqrt(4*m**4 - 4*m**2 + 8*m + 1)/4 - S(1)/4,
        m**2/2 + sqrt(4*m**4 - 4*m**2 + 8*m + 1)/4 - S(1)/4]


@XFAIL
def test_issues_6819_6820_6821_6248():
    # issue 6821
    x, y = symbols('x y', real=True)
    assert solve_univariate_real(abs(x + 3) - 2*abs(x - 3), x) == [1, 9]
    assert solve_univariate_real([abs(x) - 2, arg(x) - pi], x) == [(-2,), (2,)]
    assert set(solve_univariate_real(abs(x - 7) - 8)) == set([-S(1), S(15)])

    # issue 7145
    assert solve_univariate_real(2*abs(x) - abs(x - 1), x) == [-1, Rational(1, 3)]

    x = symbols('x')
    assert solve_univariate_real([re(x) - 1, im(x) - 2], x) == [
        {re(x): 1, x: 1 + 2*I, im(x): 2}]

    # check for 'dict' handling of solution
    eq = sqrt(re(x)**2 + im(x)**2) - 3
    assert solve_univariate_real(eq) == solve_univariate_real(eq, x)

    w = symbols('w', integer=True)
    assert solve_univariate_real(2*x**w - 4*y**w, w) == solve_univariate_real((x/y)**w - 2, w)

    # issue 2642
    assert solve_univariate_real(x*(1 + I), x) == [0]

    assert solve_univariate_real(log(x + 1) - log(2*x - 1)) == [2]

    x = symbols('x')
    assert solve_univariate_real(2**x + 4**x) == [I*pi/log(2)]


@XFAIL
def test_issue_6989():
    f = Function('f')
    assert solve_univariate_real(Eq(-f(x), Piecewise((1, x > 0), (0, True))), f(x)) == \
        [Piecewise((-1, x > 0), (0, True))]


@XFAIL
def test_lambert_multivariate():
    from sympy.abc import a, x, y
    from sympy.solve_univariate_realrs.bivariate import _filtered_gens, _lambert, _solve_univariate_real_lambert

    assert _filtered_gens(Poly(x + 1/x + exp(x) + y), x) == set([x, exp(x)])
    assert _lambert(x, x) == []
    assert solve_univariate_real((x**2 - 2*x + 1).subs(x, log(x) + 3*x)) == [LambertW(3*S.Exp1)/3]
    assert solve_univariate_real((x**2 - 2*x + 1).subs(x, (log(x) + 3*x)**2 - 1)) == \
          [LambertW(3*exp(-sqrt(2)))/3, LambertW(3*exp(sqrt(2)))/3]
    assert solve_univariate_real((x**2 - 2*x - 2).subs(x, log(x) + 3*x)) == \
          [LambertW(3*exp(1 + sqrt(3)))/3, LambertW(3*exp(-sqrt(3) + 1))/3]
    assert solve_univariate_real(x*log(x) + 3*x + 1, x) == [exp(-3 + LambertW(-exp(3)))]
    eq = (x*exp(x) - 3).subs(x, x*exp(x))
    assert solve_univariate_real(eq) == [LambertW(3*exp(-LambertW(3)))]
    # coverage test
    raises(NotImplementedError, lambda: solve_univariate_real(x - sin(x)*log(y - x), x))

    # if sign is unknown then only this one solution is obtained
    assert solve_univariate_real(3*log(a**(3*x + 5)) + a**(3*x + 5), x) == [
        -((log(a**5) + LambertW(S(1)/3))/(3*log(a)))]  # tested numerically
    p = symbols('p', positive=True)
    assert solve_univariate_real(3*log(p**(3*x + 5)) + p**(3*x + 5), x) == [
        log((-3**(S(1)/3) - 3**(S(5)/6)*I)*LambertW(S(1)/3)**(S(1)/3)/(2*p**(S(5)/3)))/log(p),
        log((-3**(S(1)/3) + 3**(S(5)/6)*I)*LambertW(S(1)/3)**(S(1)/3)/(2*p**(S(5)/3)))/log(p),
        log((3*LambertW(S(1)/3)/p**5)**(1/(3*log(p)))),]  # checked numerically
    # check collection
    assert solve_univariate_real(3*log(a**(3*x + 5)) + b*log(a**(3*x + 5)) + a**(3*x + 5), x) == [
        -((log(a**5) + LambertW(1/(b + 3)))/(3*log(a)))]

    eq = 4*2**(2*p + 3) - 2*p - 3
    assert _solve_univariate_real_lambert(eq, p, _filtered_gens(Poly(eq), p)) == [
        -S(3)/2 - LambertW(-4*log(2))/(2*log(2))]

    # issue 4271
    assert solve_univariate_real((a/x + exp(x/2)).diff(x, 2), x) == [
        6*LambertW((-1)**(S(1)/3)*a**(S(1)/3)/3)]

    assert solve_univariate_real((log(x) + x).subs(x, x**2 + 1)) == [
        -I*sqrt(-LambertW(1) + 1), sqrt(-1 + LambertW(1))]

    # these only give one of the solutions (see XFAIL below)
    assert solve_univariate_real(x**3 - 3**x, x) == [-3/log(3)*LambertW(-log(3)/3)]
    #     replacing 3 with 2 in the above solution gives 2
    assert solve_univariate_real(x**2 - 2**x, x) == [2]
    assert solve_univariate_real(-x**2 + 2**x, x) == [2]
    assert solve_univariate_real(3**cos(x) - cos(x)**3) == [
        acos(-3*LambertW(-log(3)/3)/log(3))]


@XFAIL
def test_other_lambert():
    from sympy.abc import x
    assert solve_univariate_real(3*sin(x) - x*sin(3), x) == [3]
    assert set(solve_univariate_real(3*log(x) - x*log(3))) == set(
        [3, -3*LambertW(-log(3)/3)/log(3)])
    a = S(6)/5
    assert set(solve_univariate_real(x**a - a**x)) == set(
        [a, -a*LambertW(-log(a)/a)/log(a)])
    assert set(solve_univariate_real(3**cos(x) - cos(x)**3)) == set(
        [acos(3), acos(-3*LambertW(-log(3)/3)/log(3))])
    assert set(solve_univariate_real(x**2 - 2**x)) == set(
        [2, -2/log(2)*LambertW(log(2)/2)])


@XFAIL
def test_rewrite_trig():
    assert solve_univariate_real(sin(x) + tan(x)) == [0, 2*pi]
    assert solve_univariate_real(sin(x) + sec(x)) == [
        -2*atan(-S.Half + sqrt(2 - 2*sqrt(3)*I)/2 + sqrt(3)*I/2),
        2*atan(S.Half - sqrt(3)*I/2 + sqrt(2 - 2*sqrt(3)*I)/2),
        2*atan(S.Half - sqrt(2 + 2*sqrt(3)*I)/2 + sqrt(3)*I/2),
        2*atan(S.Half + sqrt(2 + 2*sqrt(3)*I)/2 + sqrt(3)*I/2)]
    assert solve_univariate_real(sinh(x) + tanh(x)) == [0, I*pi]


@XFAIL
def test_rewrite_trigh():
    # if this import passes then the test below should also pass
    from sympy import sech
    assert solve_univariate_real(sinh(x) + sech(x)) == [
        2*atanh(-S.Half + sqrt(5)/2 - sqrt(-2*sqrt(5) + 2)/2),
        2*atanh(-S.Half + sqrt(5)/2 + sqrt(-2*sqrt(5) + 2)/2),
        2*atanh(-sqrt(5)/2 - S.Half + sqrt(2 + 2*sqrt(5))/2),
        2*atanh(-sqrt(2 + 2*sqrt(5))/2 - sqrt(5)/2 - S.Half)]


@XFAIL
def test_uselogcombine():
    eq = z - log(x) + log(y/(x*(-1 + y**2/x**2)))
    assert solve_univariate_real(eq, x, force=True) == [-sqrt(y*(y - exp(z))), sqrt(y*(y - exp(z)))]
    assert solve_univariate_real(log(x + 3) + log(1 + 3/x) - 3) == [
        -3 + sqrt(-12 + exp(3))*exp(S(3)/2)/2 + exp(3)/2,
        -sqrt(-12 + exp(3))*exp(S(3)/2)/2 - 3 + exp(3)/2]


def test_atan2():
    assert solve_univariate_real(atan2(x, 2) - pi/3, x) == [2*sqrt(3)]


def test_errorinverses():
    assert solve_univariate_real(erf(x) - S.One/2, x)==[erfinv(S.One/2)]
    assert solve_univariate_real(erf(x) - S.One*2, x)==[]

    assert solve_univariate_real(erfinv(x) - 2, x)==[erf(2)]

    assert solve_univariate_real(erfc(x) - S.One, x)==[erfcinv(S.One)]
    assert solve_univariate_real(erfc(x) - S.One*3, x)==[]

    assert solve_univariate_real(erfcinv(x) - 2,x)==[erfc(2)]


@XFAIL
def test_issue_2725():
    R = Symbol('R')
    eq = sqrt(2)*R*sqrt(1/(R + 1)) + (R + 1)*(sqrt(2)*sqrt(1/(R + 1)) - 1)
    sol = solve_univariate_real(eq, R, set=True)[1]
    assert sol == set([(S(5)/3 + 40/(3*(251 + 3*sqrt(111)*I)**(S(1)/3)) +
                       (251 + 3*sqrt(111)*I)**(S(1)/3)/3,), ((-160 + (1 +
                       sqrt(3)*I)*(10 - (1 + sqrt(3)*I)*(251 +
                       3*sqrt(111)*I)**(S(1)/3))*(251 +
                       3*sqrt(111)*I)**(S(1)/3))/Mul(6, (1 +
                       sqrt(3)*I), (251 + 3*sqrt(111)*I)**(S(1)/3),
                       evaluate=False),)])


def test_piecewise():
    assert set(solve_univariate_real(Piecewise((x - 2, Gt(x, 2)), (2 - x, True)) - 3, x)) == set([-1, 5])


@XFAIL
def test_real_imag_splitting():
    a, b = symbols('a b', real=True)
    assert solve_univariate_real(sqrt(a**2 + b**2) - 3, a) == \
        [-sqrt(-b**2 + 9), sqrt(-b**2 + 9)]
    a, b = symbols('a b', imaginary=True)
    assert solve_univariate_real(sqrt(a**2 + b**2) - 3, a) == []


@XFAIL
def test_issue_7110():
    y = -2*x**3 + 4*x**2 - 2*x + 5
    assert any(ask(Q.real(i)) for i in solve_univariate_real(y))

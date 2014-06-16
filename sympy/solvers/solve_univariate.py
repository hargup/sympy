from __future__ import print_function, division

from sympy.core.sympify import sympify
from sympy.core import S, Pow, Dummy, pi, C
from sympy.core.compatibility import (ordered)
from sympy.core.numbers import oo, zoo
from sympy.core.containers import Dict
from sympy.core.function import Lambda

from sympy.simplify.simplify import simplify, fraction

from sympy.functions import (log, Abs, tan, atan)
from sympy.sets import Interval, FiniteSet, EmptySet, imageset, Union

from sympy.polys import (roots, Poly, degree, together)

from sympy.solvers.solvers import checksol, denoms

from sympy.utilities.iterables import flatten

def invert(f, x, y=None):
    y = y or Dummy('y')
    return _invert(f, x).subs(x, y)


def _invert(f, symbol):
    """
    Returns the list of the inverse function for given real valued function

    Examples
    =========

    >>> from sympy.solvers.solve_univariate_real import _invert
    >>> from sympy.abc import x
    >>> from sympy import Abs, exp
    >>> _invert(Abs(x), x)
    [-x, x]
    >>> _invert(exp(x), x)
    [log(x)]
    """
    # XXX: there is already a _invert function in the namespace in the
    # polynomials module be careful.
    # We might dispach it into the functions themselves
    if not f.has(symbol):
        raise ValueError("Inverse of constant function doesn't exist")

    if f.is_Symbol:
        return FiniteSet(f)

    if hasattr(f, 'inverse') and not isinstance(f, C.TrigonometricFunction):
        return _invert(f.args[0], symbol).subs(symbol, f.inverse()(symbol))

    if isinstance(f, Abs):
        return FiniteSet(-f.args[0], f.args[0])

    if f.is_Mul:
        # f = g*h
        g, h = f.as_independent(symbol)

        # Maybe we can add the logic for lambert pattern here, better
        # create a different function for it.
        if g != S.One:
            return _invert(h, symbol).subs(symbol, symbol/g)

    if f.is_Add:
        # f = g + h
        g, h = f.as_independent(symbol)
        if g != S.Zero:
            return _invert(h, symbol).subs(symbol, symbol - g)

    if f.is_Pow:
        base, expo = f.args
        base_has_sym = base.has(symbol)
        expo_has_sym = expo.has(symbol)

        if not expo_has_sym:
            res = _invert(base, symbol).subs(symbol, Pow(symbol, 1/expo))
            if expo.is_rational:
                numer, denom = expo.as_numer_denom()
                if numer == S.One or numer == - S.One:
                    return res
                else:
                    if numer %2 == 0:
                        n = Dummy()
                        neg_res = imageset(Lambda(n, -n), res)
                        return res + neg_res
                    else:
                        return res
            else:
                if not base.is_positive:
                    raise ValueError("x**w where w is irrational is not defined"
                                     " for negative x")
                return res

        if not base_has_sym:
            return _invert(expo, symbol).subs(symbol, log(symbol)/log(base))

    if isinstance(f, tan):
        # something like tan(sin(x)) as f won't work, identify the case and raise a sane error
        n = Dummy()
        if isinstance(_invert(f.args[0], symbol), FiniteSet):
            return Union(*[imageset(Lambda(symbol, invt),
                                    imageset(Lambda(n, n*pi + atan(symbol)), S.Integers))
                           for invt in _invert(f.args[0], symbol)])
        else:
            raise NotImplementedError


    raise NotImplementedError


def domain_check(f, symbol, p):
    """
    Verifies if the point p is in the domain of f by checking any of
    the subexpression doesn't go unbounded.

    Examples
    ========

    >>> from sympy import domain_check
    >>> from sympy.abc import x
    >>> g = 1/(1 + (1/(x+1))**2)
    >>> domain_check(g, x, -1)
    False
    >>> domain_check(x**2, x, 0)
    True
    """
    # This function relies on the assumption that the original form of
    # the equation has not changed. A clear fail example is x/x, `x = 0` is not
    # in the domain of the function if we are working in real number system, here x/x
    # is not same as 1. Other example is 1/(1/(x+1))**2 is automatically simplified to
    # (x+1)**2. One way to tackle this can be disallowing such erroneous simplifications.
    # But not simplifying x/x to 1 doesn't feel like a good idea. The other convinient way
    # can be operating in the extended number system where -oo, oo and zoo are treated as
    # entinties. OK, extended reals won't solve the problem as x/x is still not one here,
    # and then we won't be able to simplify x - x to 0 too, because oo - oo != 0. Yes, so maybe
    # disallowing such automatic simplification is the only option
    if f.is_Atom:
        return True
    else:
        if f.subs(symbol, p).is_unbounded:
            return False
        else:
            return all([domain_check(arg, symbol, p)
                        for arg in f.args])


def solve_univariate_real(f, symbol):
    """
    real valued univariate equation solver.
    The function assums all the symbols are real.

    Examples
    ========

    >>> from sympy import solve_univariate_real
    >>> x = Symbol('x', real=True)
    >>> solve_univariate_real(x**2 - 1, x)
    {1, -1}
    """

    original_eq = f
    f = sympify(f)
    f = simplify(f)
    result = EmptySet()

    if not f.has(symbol):
        return EmptySet()
    elif f.is_Mul:
        result = FiniteSet(*flatten([list(solve_univariate_real(m, symbol)) for m in f.args]))
    elif f.is_Function:
        if f.is_Piecewise:
            result = EmptySet()
            expr_set_pairs = f.as_expr_set_pairs()
            for (expr, in_set) in expr_set_pairs:
                solns = [s for s in solve_univariate_real(expr, symbol)
                         if s in in_set]
                result = result + FiniteSet(solns)
        else:
            v = Dummy()
            inversion = invert(f, symbol, v)
            result = FiniteSet(*[i.subs({v: 0}) for i in inversion])
    else:
        f = together(f, deep=True)
        g, h = fraction(f)
        if not h.has(symbol):
            result = solve_as_poly(g, symbol)
        else:
            result = solve_univariate_real(g, symbol) - \
                    solve_univariate_real(h, symbol)

    result = [s for s in result if s.is_bounded is not False and s.is_real is True
              and domain_check(original_eq, symbol, s)]
    return FiniteSet(result)


def solve_as_poly(f, symbol):
    """
    Solve the equation using techniques of solving polynomial equations.
    That included both the polynomial equations and the equations that
    can be converted to polynomial.
    """

    if f.is_polynomial(symbol):
        solns = roots(f, symbol, cubics=True, quartics=True, quintics=True)
        num_roots = sum(solns.values())
        if degree(f, symbol) == num_roots:
            return FiniteSet(*solns.keys())
        else:
            raise ValueError("Sympy couldn't find all the roots of the "
                             "equation %s" % f)
    elif not f.is_Function and not f.is_Mul:
        # These conditions are taken care off in solve_univariate_real
        poly = Poly(f)
        if poly is None:
            raise ValueError('could not convert %s to Poly' % f)
        gens = [g for g in poly.gens if g.has(symbol)]

        if len(gens) == 1:
            poly = Poly(poly, gens[0])
            gen = poly.gen
            deg = poly.degree()
            poly = Poly(poly.as_expr(), poly.gen, composite=True)
            soln = FiniteSet(*roots(poly, cubics=True, quartics=True,
                              quintics=True).keys())

            if len(soln) < deg:
                raise ValueError('Couldn\'t find all the roots of'
                                 'the equation %s' % f)
            if gen != symbol:
                u = Dummy()
                v = Dummy()
                inversion = invert(gen - u, symbol, v)
                soln = FiniteSet(*[i.subs({u: s, v: 0}) for i in
                                         inversion for s in soln])
            result = soln
            return FiniteSet(result)
        else:
            raise NotImplementedError
    else:
        return solve_univariate_real(f, symbol)
    raise NotImplementedError


def solve_univariate_complex(f, symbol):
    """
    Solves the given univariate equation where the variable
    is a complex number
    """
    if f.is_polynomial(symbol):
        solns = roots(f, symbol, cubics=True, quartics=True, quintics=True)
        no_roots = sum(solns.values())
        if degree(f, symbol) == no_roots:
            return FiniteSet(*solns.keys())
        else:
            raise ValueError("Sympy couldn't find all the roots of the "
                             "equation %s" % f)
    else:
         f = together(f, deep=True)
         g, h = fraction(f)
         if g.is_polynomial(symbol) and h.is_polynomial(symbol):
             result = solve_univariate_complex(g, symbol) - solve_univariate_complex(h, symbol)
             return result
    raise NotImplementedError('The algorithms for solving %s are implemened' % f)

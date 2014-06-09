from __future__ import print_function, division

from sympy.core.sympify import sympify
from sympy.core import S, Pow, Dummy, pi, C
from sympy.core.compatibility import (ordered)
from sympy.core.numbers import oo, zoo
from sympy.core.containers import Dict

from sympy.simplify.simplify import simplify, fraction

from sympy.functions import (log, Abs)
from sympy.sets import Interval

from sympy.polys import (roots, Poly, degree, together)

from sympy.solvers.solvers import checksol, denoms

from sympy.utilities.iterables import flatten

def invert(f, x, y=None):
    y = y or Dummy('y')
    return [i.subs(x, y) for i in _invert(f, x)]


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
        return [f]

    if hasattr(f, 'inverse'):
        return [invt.subs(symbol, (f.inverse())(symbol)) for invt
                in _invert(f.args[0], symbol)]

    if isinstance(f, Abs):
        return [-f.args[0], f.args[0]]

    if f.is_Mul:
        # f = g*h
        g, h = f.as_independent(symbol)

        # Maybe we can add the logic for lambert pattern here, better
        # create a different function for it.
        if g != S.One:
            return [invt.subs(symbol, symbol / g)
                    for invt in _invert(h, symbol)]

    if f.is_Add:
        # f = g + h
        g, h = f.as_independent(symbol)
        if g != S.Zero:
            return [invt.subs(symbol, symbol - g)
                    for invt in _invert(h, symbol)]

    if f.is_Pow:
        base, expo = f.args
        base_has_sym = base.has(symbol)
        expo_has_sym = expo.has(symbol)

        if not expo_has_sym:
            if expo.is_rational:
                numer, denom = expo.as_numer_denom()
                if numer == S.One or numer == - S.One:
                    return [invt.subs(symbol, Pow(symbol, 1/expo)) for invt
                            in _invert(base, symbol)]
                else:
                    pos_res = [invt.subs(symbol, Pow(symbol, 1/expo)) for invt
                               in _invert(base, symbol)]

                    if numer %2 == 0:
                        neg_res = [-invt.subs(symbol, Pow(symbol, 1/expo)) for invt
                                   in _invert(base, symbol)]
                        return pos_res + neg_res
                    else:
                        return pos_res
            else:
                if not base.is_positive:
                    raise ValueError("x**w where w is irrational is not defined"
                                     " for negative x")
                return [invt.subs(symbol, Pow(symbol, 1/expo)) for invt
                        in _invert(base, symbol)]

        if not base_has_sym:
            return [invt.subs(symbol, log(symbol)/log(base)) for invt
                    in _invert(expo, symbol)]


    raise NotImplementedError


def subexpression_checking(f, symbol, p):
    """
    Verifies if the point p is in the domain of f by checking of none of
    the subexpression attains value infinity.

    Examples
    ========

    >>> from sympy import subexpression_checking
    >>> from sympy.abc import x
    >>> g = 1/(1 + (1/(x+1))**2)
    >>> subexpression_checking(g, x, -1)
    False
    >>> subexpression_checking(x**2, x, 0)
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
            return all([subexpression_checking(arg, symbol, p)
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
    [1, -1]
    """

    original_eq = f
    f = sympify(f)
    f = simplify(f)
    result = set()

    if not f.has(symbol):
        result = set()
    elif f.is_Mul:
        result = set(flatten([solve_univariate_real(m, symbol) for m in f.args]))
    elif f.is_Function:
        if f.is_Piecewise:
            result = set()
            expr_set_pairs = f.as_expr_set_pairs()
            for (expr, in_set) in expr_set_pairs:
                solns = [s for s in solve_univariate_real(expr, symbol)
                         if s in in_set]
                result.update(set(solns))
        else:
            v = Dummy()
            inversion = invert(f, symbol, v)
            result = set([i.subs({v: 0}) for i in inversion])
    else:
        f = together(f, deep=True)
        g, h = fraction(f)
        if not h.has(symbol):
            result = solve_as_poly(g, symbol)
        else:
            result = set(solve_univariate_real(g, symbol)) - set(solve_univariate_real(h, symbol))

    result = [s for s in result if s not in [-oo, oo, zoo] and s.is_real is True
              and subexpression_checking(original_eq, symbol, s)]
    return result


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
            return list(solns.keys())
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
            soln = list(roots(poly, cubics=True, quartics=True,
                              quintics=True).keys())

            if len(soln) < deg:
                raise ValueError('Couldn\'t find all the roots of'
                                 'the equation %s' % f)
            if gen != symbol:
                u = Dummy()
                v = Dummy()
                inversion = invert(gen - u, symbol, v)
                soln = list(ordered(set([i.subs({u: s, v: 0}) for i in
                                         inversion for s in soln])))
            result = soln
            return result
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
            return list(solns.keys())
        else:
            raise ValueError("Sympy couldn't find all the roots of the "
                             "equation %s" % f)
    else:
         f = together(f, deep=True)
         g, h = fraction(f)
         if g.is_polynomial(symbol) and h.is_polynomial(symbol):
             result = set(solve_univariate_complex(g, symbol)) - set(solve_univariate_complex(h, symbol))
             return list(result)
    raise NotImplementedError('The algorithms for solving %s are implemened' % f)

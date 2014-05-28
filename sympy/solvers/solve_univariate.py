from __future__ import print_function, division

from sympy.functions import (log, exp, Abs)
from sympy.polys import (roots, Poly, degree)
from sympy.core import S, Pow, Dummy
from sympy.core.compatibility import (ordered)


def invert(f, x, y=None):
    y = y or Dummy('y')
    return [i.subs(x, y) for i in _invert(f, x)]


def _invert(f, symbol):
    """
    Returns the list of the inverse function

    Examples
    =========

    >>> from sympy.solvers.solve_univariate import _invert
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
    if f.is_Symbol:
        return [f]
    if isinstance(f, exp):
        return [invt.subs(symbol, log(symbol)) for invt
                in _invert(f.args[0], symbol)]
    if isinstance(f, log):
        return [invt.subs(symbol, exp(symbol)) for invt
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

    raise NotImplementedError


def solve_univariate(f, symbol):
    """
    univariate equation solver

    Examples
    ========

    >>> from sympy import solve_univariate
    >>> from sympy.abc import x
    >>> solve_univariate(x**2 - 1, x)
    [1, -1]
    """
    result = set()
    if f.is_Mul:
        result.update(*[solve_univariate(m, symbol) for m in f.args])
        return list(result)
    elif f.is_Function:
        if f.is_Piecewise:
            # XXX: Piecewise functions in sympy are implicitly real,
            # the conditionals greaterthan, lessthan are not defined
            # for complex valued functions do something for it.
            # also ploting for the piecewise functions doesn't work,
            # it wll be easy to implement. Create a issue for it.
            # TODO: write the docstring for the as_expr_set_pairs method for
            # piecewise functions
            result = set()
            expr_set_pairs = f.as_expr_set_pairs()
            for (expr, in_set) in expr_set_pairs:
                solns = [s for s in solve_univariate(expr, symbol)
                         if s in in_set]
                result.update(set(solns))
            return list(result)
        else:
            raise NotImplementedError
    else:
        return solve_as_poly(f, symbol)


def solve_as_poly_gen_is_pow(poly):
    """ Solve a polynomial equation where the generator is of form x**m """
    # TODO: All of the logic of this function should be better placed in invert
    # It will also simplify te code of solve_as_poly

    expo = poly.gen.args[1]
    if expo.is_Rational:
        numer, denom = expo.as_numer_denom()
        solns = roots(poly, poly.gen)
        if len(solns) < poly.degree():
            raise ValueError("Sympy couldn't evaluate all the "
                             "roots of the polynomial %s" % poly)
        if numer is S.One:
            return [Pow(sol, denom) for sol
                    in roots(poly, poly.gen)]
        elif numer is - S.One:
            return [1/Pow(sol, denom) for sol
                    in roots(poly, poly.gen) if not sol == S.Zero]
        else:
            # This case shouldn't arise. Why?
            # x**(2/3) should not be a generator rather, the generator
            # shall be x**(1/3) and the 2 of the numerator should add to
            # the degree of the polynomial
            raise NotImplementedError
    elif expo.is_Real and not expo.if_Rational:
        return NotImplementedError("x**w = c have infinitely many"
                                   " solutions if w is irrational")
    elif not expo.is_Real:
        # See Fateman's paper
        # http://www.cs.berkeley.edu/~fateman/papers/y=z2w.pdf
        # For the solution for z**(w) = y,
        # where w is real
        # z = y**(1/w) for w >= 1
        # for 0 <= w < 1 y can only take the values where the
        # argument theta follows the condition -w*pi < theta <= w*pi,
        # these conditions cannot be incoperated in the invert function,
        # and we need the value of y in order to solve the equation.
        # The conditions where w is complex are more complicated.
        # XXX, TODO: we have to fix the bug in powsimp to not to simplify
        # (z**(1/w))**w to y, it works fine if w is a symbol, else it
        # doesn't e.g., we are wrongly simplifying
        # y - (y**(1/(1 + I)))**(1 + I) as 0
        return NotImplementedError


def solve_as_poly(f, symbol):
    """
    Solves the equation as polynomial or converting
    it to a polynomial.
    """

    if f.is_polynomial(symbol):
        solns = roots(f, symbol)
        no_roots = sum(solns.values())
        if degree(f, symbol) == no_roots:
            return list(solns.keys())
        else:
            raise NotImplementedError
    elif not f.is_Function and not f.is_Mul:
        # These conditions are taken care off in solve_univariate
        poly = Poly(f)
        if poly is None:
            raise ValueError('could not convert %s to Poly' % f)
            # Shouldn't it be NotImplementedError
        gens = [g for g in poly.gens if g.has(symbol)]

        if len(gens) == 1:
            poly = Poly(poly, gens[0])
            gen = poly.gen
            if poly.gen.is_Pow:
                # XXX: clean this up, we don't need it.
                soln = solve_as_poly_gen_is_pow(poly)
                gen_base = gen.args[0]
                if gen_base != symbol:
                    u = Dummy()
                    v = Dummy()
                    inversion = invert(gen - u, symbol, v)
                    soln = list(ordered(set([i.subs({u: s, v: 0}) for i in
                                             inversion for s in soln])))
            else:
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
        return solve_univariate(f, symbol)
    raise NotImplementedError

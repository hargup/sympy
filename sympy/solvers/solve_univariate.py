from __future__ import print_function, division

from sympy.functions import (log, exp)
from sympy.polys import (roots, Poly, degree)
from sympy.core import S, Pow, Dummy, ordered


def invert(f, symbol):
    """ Returns the list of the inverse function """
    # We might dispach it into the functions themselves
    if f.is_Symbol:
        return [f]
    if isinstance(f, exp):
        return [invt.subs(symbol, log(symbol)) for invt
                in invert(f.args[0], symbol)]
    if isinstance(f, log):
        return [invt.subs(symbol, exp(symbol)) for invt
                in invert(f.args[0], symbol)]
    if f.is_Mul:
        # f = g*h
        g, h = f.as_independent(symbol)

        # Maybe we can add the logic for lambert pattern here, better
        # create a different function for it.
        if g != S.One:
            return [invt/g for invt in invert(h, symbol)]
    if f.is_Add:
        # f = g*h
        g, h = f.as_independent(symbol)
        if g != S.Zero:
            return [invt - g for invt in invert(h, symbol)]

    raise NotImplementedError


def solve_univariate(f, symbol):
    """ A simple and robust implementation for univariate solvers """
# 23 May
# The current approach was to read the _solve function, understand some parts
# copy the code here with necessary modifications to ensure that it is robust.
# The current _solve function even for the univariate equations is quite big
# and complicated. It is a single function with around 300 lines.
# It is not easy to understant it, even if it was I feel I should rewrite
# the solver from the scratch, because the main motive of the refactoring
# is to avoid the mistakes the of the current _solve, and current approach
# is making it look a lot like the old code, maybe I'm risking reinventing
# a lot of thing but prehaps it is worth it. And it isn't that much of a risk
# because I have read the code so there won't be much intellectual reinvention
# of wheels. Using the code has risk of repeating the old mistakes. Maybe I'll
# come up with some novel methods.

# XXX: solve_univariate cannot accept input in form of Eq(f, 0)
# this won't be handled here, everything related to the interface will
# be handled in solve

    if f.is_polynomial(symbol):
        solns = roots(f, symbol)
        no_roots = sum([solns[key] for key in solns.keys()])
        if degree(f, symbol) == no_roots:
            return list(solns.keys())
        else:
            raise NotImplementedError("")  # TODO: write the error message

    # XXX: not doing the unrad from the _solve due to lack of knowledge of the
    # internals of the function
    # TODO: read the implementation of unrad

    elif f.is_Mul:
        result = set()
        for m in f.args:
            solns = solve_univariate(m, symbol)
            result.update(set(solns))
        return list(result)

    elif f.is_Piecewise:
# 22 May 2014
# XXX: Piecewise functions in sympy are implicitly real,
# the conditionals greaterthan, lessthan are not defined
# for complex valued functions do something for it.
# also ploting for the piecewise functions doesn't work, it wll be easy
# to implement create a issue for it.
# TODO: write the docstring for the as_expr_set_pairs method for
# piecewise functions
        result = set()
        expr_set_pairs = f.as_expr_set_pairs()
        for (expr, in_set) in expr_set_pairs:
            solns = [s for s in solve_univariate(expr, symbol)
                     if s in in_set]
            result.update(set(solns))
        return list(result)

    # Transform the expression to polynomial
    else:
        poly = Poly(f)
        if poly is None:
            raise ValueError('could not convert %s to Poly' % f)
            # Shouldn't it be NotImplementedError
        gens = [g for g in poly.gens if g.has(symbol)]

        if len(gens) == 1:
# There is only one generator that we are interested in, but there
# may have been more than one generator identified by polys (e.g.
# for symbols other than the one we are interested in) so recast
# the poly in terms of our generator of interest.

# TODO: break it off into smaller different function
# A large monolithic function isn't a good thing

            if len(poly.gens) > 1:
                poly = Poly(poly, gens[0])
                if poly.degree() == 1 and (
                        poly.gen.is_Pow and
                        poly.gen.is_Rational):
 # 23rd May
 # Maybe we can create a different function invert here
 # to handle different branches of the inverse functions
 # The condition for Rational isn't is also good.

 # 24th May
 # See Fateman's paper http://www.cs.berkeley.edu/~fateman/papers/y=z2w.pdf
 # For the solution for z**(w) = y,
 # where w is real
 # z = y**(1/w) for w >= 1
 # for 0 <= w < 1 y can only take the values where the argument theta follows
 # the condition -w*pi < theta <= w*pi,
 # these conditions cannot be incoperated in the invert function, and we need
 # the value of y in order to solve the equation.
 # The conditions where w is complex are more complicated.
 # XXX, TODO: we have to fix the bug in powsimp to not to simplify
 # (z**(1/w))**w to y, it works fine if w is a symbol, else it doesn't
 # e.g., we are wrongly simplifying y - (y**(1/(1 + I)))**(1 + I) as 0
                    expo = poly.gen.args[1]
                    numer, denom = expo.as_numer_denom()
                    return [Pow(sol, denom) for sol in roots(poly)]
                elif poly.degree() == 1 and \
                        (poly.gen.is_Pow and not poly.gen.is_Rational):
                    return NotImplementedError("x**w = c have infinitely many"
                                               " solutions if w is irrational")
                else:
                    deg = poly.degree()
                    poly = Poly(poly.as_expr(), poly.gen, composite=True)
                    soln = list(roots(poly, cubics=True, quartics=True,
                                      quintics=True).keys())

                    if len(soln) < deg:
                        raise ValueError('Couldn\'t find all the roots of'
                                         'the equation %s' % f)
                    gen = poly.gen
                    if gen != symbol:
                        u = Dummy()
                        inversion = solve_univariate(gen - u, symbol)
                        soln = list(ordered(set([i.subs(u, s) for i in
                                    inversion for s in soln])))
                    result = soln
        else:
            raise NotImplementedError('There were more than one generator'
                                      'while converting to polynomial')


def solve_as_poly(f, symbol):
    if f.is_polynomial(symbol):
        solns = roots(f, symbol)
        no_roots = sum([solns[key] for key in solns.keys()])
        if degree(f, symbol) == no_roots:
            return list(solns.keys())
        else:
            raise NotImplementedError("")  # TODO: write the error message
    elif not f.is_Funtion and not f.is_Mul:
        poly = Poly(f)
        if poly is None:
            raise ValueError('could not convert %s to Poly' % f)
            # Shouldn't it be NotImplementedError
        gens = [g for g in poly.gens if g.has(symbol)]

        if len(gens) == 1:
            if len(poly.gens) > 1:
                poly = Poly(poly, gens[0])
                if poly.degree() == 1 and (
                        poly.gen.is_Pow and
                        poly.gen.is_Rational):
                    expo = poly.gen.args[1]
                    numer, denom = expo.as_numer_denom()
                    return [Pow(sol, denom) for sol in roots(poly)]
                elif poly.degree() == 1 and \
                        (poly.gen.is_Pow and not poly.gen.is_Rational):
                    return NotImplementedError("x**w = c have infinitely many"
                                               " solutions if w is irrational")
                else:
                    deg = poly.degree()
                    poly = Poly(poly.as_expr(), poly.gen, composite=True)
                    soln = list(roots(poly, cubics=True, quartics=True,
                                      quintics=True).keys())

                    if len(soln) < deg:
                        raise ValueError('Couldn\'t find all the roots of'
                                         'the equation %s' % f)
                    gen = poly.gen
                    if gen != symbol:
                        u = Dummy()
                        inversion = solve_univariate(gen - u, symbol)
                        soln = list(ordered(set([i.subs(u, s) for i in
                                    inversion for s in soln])))
                    result = soln
                    return result
    raise NotImplementedError

from sympy import Wild, solve, simplify, log, exp, \
    sin, cos, tan, cot, sec, csc, asin, acos, atan, acot, \
    limit, oo


def infinite_discontinuties(expr, sym):
    """
    Find the points of infinite discontinuities of a real univariate function

    Example
    =========

    >>> from sympy.calculus.singularities import infinite_discontinuties
    >>> from sympy import exp, log, Symbol
    >>> x = Symbol('x', real=True)
    >>> infinite_discontinuties(log((x-2)**2) + x**2, x)
    [2]
    >>> infinite_discontinuties(exp(1/(x-2)**2), x)
    [2]

    """
    def _has_piecewise(e):
        if e.is_Piecewise:
            return e.has(sym)
        return any([_has_piecewise(a) for a in e.args])

    if _has_piecewise(expr):
        raise NotImplementedError("Expressions with Piecewise functions "
                                  " are not yet implemented")

    if expr.has(sin, cos, tan, cot, sec, csc, asin, acos, atan, acot):
        # Not trying for trig function because they have infinitely
        # many solutions. Currently we don't have methods to return or
        # handle them.

        raise NotImplementedError("Expressions containing trigonometric"
                                  "functions are not supported")

    if expr.is_polynomial():
        return []

    pods = []  # pod: Points of discontinuties
    pods = pods + [x for x in solve(simplify(1/expr), sym)
                   if x.is_real]
    p = Wild("p")
    q = Wild("q")
    r = Wild("r")

    # check the condition for log
    expr_dict = expr.match(r*log(p) + q)
    if not expr_dict[r].is_zero:
        pods += solve(expr_dict[p], sym)
        pods += infinite_discontinuties(expr_dict[p], sym)
        pods += infinite_discontinuties(expr_dict[r], sym)

    # check the condition for exp
    expr = expr.rewrite(exp)
    expr_dict = expr.match(r*exp(p) + q)
    if not expr_dict[r].is_zero:
        pods += [x for x in solve(simplify(1/expr_dict[p]), sym)
                 if limit(expr_dict[p], sym, x) == oo]
        pods += [x for x in infinite_discontinuties(expr_dict[p], sym)
                 if limit(expr_dict[p], sym, x) == oo]
        pods += infinite_discontinuties(expr_dict[r], sym)

    return list(set(pods))  # remove dublications

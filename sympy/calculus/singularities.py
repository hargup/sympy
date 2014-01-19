from sympy import Wild, solve, simplify, limit, exp, log
from sympy.core import Add, Mul, Pow, oo
from sympy.utilities.iterables import flatten


def infinite_discontinuties(expr, sym):
    """
    Find the points of infinite discontinuities of a real univariate function.
    Currently only rational functions and their composition and combinations
    with exp and log functions are supported.

    Examples
    ========

    >>> from sympy.calculus.singularities import infinite_discontinuties
    >>> from sympy import exp, log, Symbol
    >>> x = Symbol('x', real=True)
    >>> infinite_discontinuties(log((x-2)**2) + x**2, x)
    [2]
    >>> infinite_discontinuties(exp(1/(x-2)**2), x)
    [2]
    >>> infinite_discontinuties(log(exp(1/(x-2)**2) + 1) + 1/(x+1), x)
    [-1, 2]

    """

    if not _has_only_supported_func(expr, sym):
        raise NotImplementedError("Sorry, algorithms to find infinite"
                                  " discontinuties of %s are not yet"
                                  " implemented" % expr)

    if expr.is_polynomial(sym):
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
        pods += [soln for soln in solve(expr_dict[p], sym) if soln.is_real]
        pods += infinite_discontinuties(expr_dict[p], sym)
        pods += infinite_discontinuties(expr_dict[r], sym)

    # check the condition for exp
    expr = expr.rewrite(exp)
    expr_dict = expr.match(r*exp(p) + q)
    if not expr_dict[r].is_zero:
        # exp(f) has infinite discontinuity only for f -> oo
        pods += [x for x in solve(simplify(1/expr_dict[p]), sym)
                 if limit(expr_dict[p], sym, x) == oo and x.is_real]
        pods += [x for x in infinite_discontinuties(expr_dict[p], sym)
                 if limit(expr_dict[p], sym, x) == oo and x.is_real]
        pods += infinite_discontinuties(expr_dict[r], sym)

    return sorted(list(set(pods)))  # remove dublications


def _basic_args(e, sym):
    """ Returns the leaves of the sympy expression tree """
    basic_arg_list = []
    if isinstance(e, Add) or isinstance(e, Mul) or isinstance(e, Pow):
        basic_arg_list += flatten([_basic_args(g, sym) for g in e.args])
    else:
        basic_arg_list += [e]
    return basic_arg_list


def _has_only_supported_func(e, sym):
    func_list = [exp, log]
    for f in _basic_args(e, sym):
        if f.is_rational_function(sym):
            return True
        elif not any([isinstance(f, func) for func in func_list]):
            return False
        else:
            return all([_has_only_supported_func(g, sym) for g in f.args])

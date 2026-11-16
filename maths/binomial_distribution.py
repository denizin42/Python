"""For more information about the Binomial Distribution:
https://en.wikipedia.org/wiki/Binomial_distribution
"""

from __future__ import annotations

from math import exp, lgamma, log, log1p


def binomial_distribution(successes: int, trials: int, prob: float) -> float:
    """
    Return probability of k successes out of n tries, with p probability for one
    success.

    Uses log-gamma and logarithms to compute the probability in log-space,
    avoiding float(factorial(trials)) overflow for large trials (n >= 171).

    >>> binomial_distribution(3, 5, 0.7)
    0.30870000000000036
    >>> binomial_distribution(2, 4, 0.5)
    0.37499999999999983
    >>> round(binomial_distribution(85, 171, 0.5), 6)
    0.06075
    >>> round(binomial_distribution(1000, 2000, 0.5), 6)
    0.017839
    >>> binomial_distribution(6, 5, 0.5)
    Traceback (most recent call last):
        ...
    ValueError: successes must be lower or equal to trials
    >>> binomial_distribution(-1, 5, 0.5)
    Traceback (most recent call last):
        ...
    ValueError: the function is defined for non-negative integers
    >>> binomial_distribution(2, 4, 1.5)
    Traceback (most recent call last):
        ...
    ValueError: prob has to be in range of 1 - 0
    """
    if successes > trials:
        raise ValueError("successes must be lower or equal to trials")
    if trials < 0 or successes < 0:
        raise ValueError("the function is defined for non-negative integers")
    if not isinstance(successes, int) or not isinstance(trials, int):
        raise TypeError("successes and trials must be integers")
    if not 0 < prob < 1:
        raise ValueError("prob has to be in range of 1 - 0")
    # Compute C(n, k) * p^k * (1 - p)^(n - k) in log-space to prevent overflow.
    log_coeff = (
        lgamma(trials + 1) - lgamma(successes + 1) - lgamma(trials - successes + 1)
    )
    log_prob = successes * log(prob) + (trials - successes) * log1p(-prob)
    return exp(log_coeff + log_prob)


if __name__ == "__main__":
    from doctest import testmod

    testmod()

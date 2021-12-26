from scipy.stats import bernoulli as sp_bernoulli
from scipy.stats import binom as sp_binomial
from scipy.stats import geom as sp_geometrical
from scipy.stats import poisson as sp_poisson


# Docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bernoulli.html
def bernoulli(prob, size):
    rand_vars_ber = sp_bernoulli.rvs(prob, size=size)
    return rand_vars_ber


def bernoulli_mean(prob):
    return prob


def bernoulli_var(prob):
    return prob * (1 - prob)


# Docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html#scipy.stats.binom
def binomial(n, prob, size):
    rand_vars_bin = sp_binomial.rvs(n, prob, size=size)
    return rand_vars_bin


def binomial_mean(n, prob):
    return n * prob


def binomial_var(n, prob):
    return prob * (1 - prob) * n


# Docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.geom.html#scipy.stats.geom
def geometric(prob, size):
    rand_vars_geo = sp_geometrical.rvs(prob, size=size)
    return rand_vars_geo


def geometric_mean(prob):
    return (1 - prob) / prob


def geometric_var(prob):
    return (1 - prob) / (prob ** 2)


# Docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html#scipy.stats.poisson
def poisson(mu, size):
    rand_vars_poi = sp_poisson.rvs(mu, size=size)
    return rand_vars_poi


def poisson_sigma(mu):
    return mu
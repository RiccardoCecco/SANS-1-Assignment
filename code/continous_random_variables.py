import numpy as np
from scipy.stats import expon as sp_expon
from scipy.stats import uniform as sp_uniform


# Docs: https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html
def uniform(low, high, size):
    rand_var_uni = np.random.uniform(low, high, size)
    return rand_var_uni


def uniform_mean(low, high):
    return (low + high) / 2


def uniform_var(low, high):
    return (high - low) ** 2 / 12


# Docs: https://numpy.org/doc/stable/reference/random/generated/numpy.random.exponential.html
def exponential(exp_lambda, size):
    rand_var_exp = np.random.exponential(exp_lambda, size=size)
    return rand_var_exp


def exponential_mean(exp_lambda):
    return 1 / exp_lambda


def exponential_var(exp_lambda):
    return 1 / (exp_lambda ** 2)


# Docs: https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
def gaussian(mu, sigma, size):
    rand_var_gauss = np.random.normal(mu, sigma, size=size)
    return rand_var_gauss

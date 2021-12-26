import discrete_random_variables as drv
import continous_random_variables as crv


N = 100
M = 1000


# For k, 0<k<N, we need to sum up Xk of every sample in the N size list and divide by N
def x_starred(distribution):
    # distribution is a list of samples of a given distribution
    # sample is a vector of M random variables
    # sums[i] = distribution[0][i] + ... + distribution[N-1][i]
    sums = [0] * M
    for sample in distribution:
        for i in range(M):
            sums[i] += sample[i]

    # In order to have X*, we need to divide by N every element of sums
    x = list()
    for s in sums:
        x.append(s / N)
    return x


# BERNOULLI DISTRIBUTION
def bernoulli():
    # Fill set with random samples
    bern = list()
    prob = 3 / 4
    for i in range(N):
        bern.append(drv.bernoulli(prob, M))

    x_starred_bern = x_starred(bern)
    return x_starred_bern


# UNIFORM DISTRIBUTION
def uniform():
    # Fill set with random samples
    uni = list()
    low, high = 0, 1
    for i in range(N):
        uni.append(crv.uniform(low, high, M))

    x_starred_uni = x_starred(uni)
    return x_starred_uni


# EXPONENTIAL DISTRIBUTION
def exponential():
    # Fill set with random samples
    exp = list()
    exp_lambda = 1
    for i in range(N):
        exp.append(crv.exponential(exp_lambda, M))

    x_starred_exp = x_starred(exp)
    return x_starred_exp


# GAUSSIAN DISTRIBUTION
def gaussian():
    # Fill set with random samples
    gauss = list()
    mu, sigma = 1, 1 ** (1 / 2)
    for i in range(N):
        gauss.append(crv.gaussian(mu, sigma, M))

    x_starred_gauss = x_starred(gauss)
    return x_starred_gauss

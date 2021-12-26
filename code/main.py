import plot
import discrete_random_variables as drv
import continous_random_variables as crv
import central_limit_theorem as clt
import law_large_numbers as lln
import multivariate_gaussian as multigauss
import subspaces_eigenvalues_and_eigenvectors as eigen
import orthogonal_symmetric_and_positive_definite_matrices as ortho
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    M = 10000
    # Plot discrete random variables
    # Bernoulli with p = p{X = 1}= 3/4
    prob = 3 / 4
    ber = drv.bernoulli(prob, M)
    ber_mu = drv.bernoulli_mean(prob)
    ber_sigma = drv.bernoulli_var(prob)
    plot.plot_mean_variance("Bernoulli discrete probability distribution", ber[1:1000], ber_mu, ber_sigma)

    # Binomial with n = 10 and p = 3/4
    n, prob = 10, 3 / 4
    binomial = drv.binomial(n, prob, M)
    bin_mu = drv.binomial_mean(n, prob)
    bin_sigma = drv.binomial_var(n, prob)
    plot.plot_mean_variance("Binomial discrete probability distribution", binomial[1:1000], bin_mu, bin_sigma)

    # Geometric with p = 3/4 (X taking values in {0,1,...}),
    prob = 3 / 4
    geo = drv.geometric(prob, M)
    geo_mu = drv.geometric_mean(prob)
    geo_sigma = drv.geometric_var(prob)
    plot.plot_mean_variance("Geometric discrete probability distribution", geo[1:1000], geo_mu, geo_sigma)

    # Poisson with λ = 1
    poi_mu = 1
    poi = drv.poisson(poi_mu, M)
    poi_sigma = drv.poisson_sigma(poi_mu)
    plot.plot_mean_variance("Poisson discrete probability distribution", poi[1:1000], poi_mu, poi_sigma)

    # Plot continuous random variables
    # Uniform with interval [0,1]
    low, high = 0, 1
    uni = crv.uniform(low, high, M)
    uni_mu = crv.uniform_mean(low, high)
    uni_sigma = crv.uniform_var(low, high)
    plot.plot_mean_variance("Uniform continuous probability distribution", uni[1:1000], uni_mu, uni_sigma)

    # Exponential with λ = 1
    exp_lambda = 1
    exp = crv.exponential(exp_lambda, M)
    exp_mu = crv.exponential_mean(exp_lambda)
    exp_sigma = crv.exponential_var(exp_lambda)
    plot.plot_mean_variance("Exponential continuous probability distribution", exp[1:1000], exp_mu, exp_sigma)

    # Gaussian μ =1, σ^2=1
    gauss_mu, gau1_sigma = 1, 1 ** (1 / 2)  # operand ** is for exp
    gau1 = crv.gaussian(gauss_mu, gau1_sigma, M)
    plot.plot_mean_variance("Gaussian σ2=1 continuous probability distribution", gau1[1:1000], gauss_mu, gau1_sigma)

    # Gaussian μ =1, σ^2=5
    gau5_sigma = 5 ** (1 / 2)  # operand ** is for exp
    gau5 = crv.gaussian(gauss_mu, gau5_sigma, M)
    plot.plot_mean_variance("Gaussian σ2=5 continuous probability distribution", gau5[1:1000], gauss_mu, gau5_sigma)

    # Plot Law Large Numbers
    lln_bern = lln.bernoulli()
    plot.plot_mean("Law of Large Numbers Bernoulli", lln_bern, ber_mu)

    lln_uni = lln.uniform()
    plot.plot_mean("Law of Large Numbers Uniform", lln_uni, uni_mu)

    lln_exp = lln.exponential()
    plot.plot_mean("Law of Large Numbers Exponential", lln_exp, exp_mu)

    lln_gauss = lln.gaussian()
    plot.plot_mean("Law of Large Numbers Gaussian", lln_gauss, gauss_mu)

    # Plot Central Limit Theorem
    # The distribution of Z_n approaches the standard normal distribution(mean 0, variance 1) as N → ∞.
    clt_ber = clt.bernulli_result()
    plot.plot_variance("Central Limit Theorem Bernoulli", clt_ber[1:1000], 0, 1)

    clt_uniform = clt.uniform_result()
    plot.plot_variance("Central Limit Theorem Uniform", clt_uniform[1:1000], 0, 1)

    clt_exp = clt.exponential_result()
    plot.plot_variance("Central Limit Theorem Exponential", clt_exp[1:1000], 0, 1)

    clt_gau = clt.exponential_result()
    plot.plot_variance("CLT Gaussian", clt_gau[1:1000], 0, 1)

    # Plot multivariate gaussian
    x, y = multigauss.samples()
    arrow1, arrow2 = multigauss.arrows()
    mean1, dir11, dir12 = arrow1
    mean2, dir21, dir22 = arrow2
    plot.plot_multivariate_gaussian(x, y, mean1, dir11, dir12, mean2, dir21, dir22)

    # Plot 6 experiment
    x, y = eigen.circle()
    plot.plot_circle_A(x, y, mean1, dir11, dir12, mean2, dir21, dir22)


    x, y = eigen.elipse_A()
    arrow1, arrow2 = eigen.arrows_A()
    mean1, dir11, dir12 = arrow1
    mean2, dir21, dir22 = arrow2
    plot.plot_elipse_A(x, y, mean1, dir11, dir12, mean2, dir21, dir22)

    x, y = eigen.elipse_B()
    arrow1, arrow2 = eigen.arrows_B()
    mean1, dir11, dir12 = arrow1
    mean2, dir21, dir22 = arrow2
    plot.plot_elipse_B(x, y, mean1, dir11, dir12, mean2, dir21, dir22)

    colA, kerA, colAt, kerAt = eigen.svdA()
    colB, kerB, colBt, kerBt = eigen.svdB()
    colC, kerC, colCt, kerCt = eigen.svdC()
    colD, kerD, colDt, kerDt = eigen.svdD()
    plot.plot_colA_KerAt(colA, kerAt)
    plot.plot_colAt_KerA(colAt, kerA)
    plot.plot_colB_KerBt(colB, kerBt)
    plot.plot_colBt_KerB(colBt, kerB)
    plot.plot_colC_KerCt(colC, kerCt)
    plot.plot_colCt_KerC(colCt, kerC)
    plot.plot_colD_KerDt(colD, kerDt)
    plot.plot_colDt_KerD(colDt, kerD)

    # Orthogonal, symmetric and positive definite matrices
    # Matrix A (circle)
    eigenvalues_a, eigenvectors_a = ortho.eigen_a()
    x = np.linspace(eigenvalues_a[1], eigenvalues_a[0], 100)
    y1 = np.sqrt(1 - np.square(x))
    y2 = -1 * y1

    arrow_a1, arrow_a2 = ortho.arrows_a()
    mean1, dir11, dir12 = arrow_a1
    mean2, dir21, dir22 = arrow_a2
    plot.plot_arrows_circle(0, 0, 1, 2, 2, mean1, dir11, dir12, mean2, dir21, dir22)

    # Matrix B (ellipse)
    eigenvalues_b, eigenvectors_b = ortho.eigen_b()
    arrow_b1, arrow_b2 = ortho.arrows_b()
    mean1, dir11, dir12 = arrow_b1
    mean2, dir21, dir22 = arrow_b2
    plot.plot_arrows_ellipse(10, 10, mean1, dir11, dir12, mean2, dir21, dir22)

    # Matrix C
    eigenvalues_c, eigenvectors_c = ortho.eigen_c()
    arrow_c1, arrow_c2 = ortho.arrows_c()
    mean1, dir11, dir12 = arrow_c1
    mean2, dir21, dir22 = arrow_c2
    plot.plot_arrows_ellipse(5, 5, mean1, dir11, dir12, mean2, dir21, dir22)
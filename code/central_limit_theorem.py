import discrete_random_variables as drv
import continous_random_variables as crv
import numpy

N = 100
M = 1000


def bernulli_result():
    bernoulli = []
    prob=3/4
    avg_x = prob
    var_x = prob*(1-prob)
    for i in range(N):
        bernoulli.append(drv.bernoulli(prob, M))
    x_starred = calculate(bernoulli, avg_x, var_x)
    return x_starred


def uniform_result():
    uniform=[]
    a = 0
    b = 1
    avg_x = (b + a)/2
    var_x = ((b-a)^2)/12
    for i in range(N):
        uniform.append(crv.uniform(a, b, M))
    x_starred = calculate(uniform, avg_x, var_x)
    return x_starred


def exponential_result():
    exponential = []
    lamb = 1
    avg_x = 1/lamb
    var_x = 1/(lamb^2)
    for i in range(N):
        exponential.append(crv.exponential(lamb, M))
    x_starred = calculate(exponential, avg_x, var_x)
    return x_starred


def gaussian_result():
    gaussian = []
    mu=1
    sigma = 1**(1/2)
    avg_x = 1
    var_x = 1
    for i in range(N):
        gaussian.append(crv.gaussian(mu, sigma, M))
    x_starred = calculate(gaussian, avg_x, var_x)
    return x_starred


def calculate(listP, avg_x, var_x):
    sum=0
    x_starred=[]
    aux= numpy.zeros(shape=(N,M))
    for i in range(0,100):
        for j in range(0,1000):
            aux[i][j]= (listP[i][j]- avg_x)/(var_x**(1/2))
    for i in range(0,1000):
        for j in range(0,100):
            sum+=aux[j][i]
        x_starred.append(sum/10)
        sum=0

    return x_starred

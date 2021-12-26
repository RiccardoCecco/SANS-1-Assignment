import matplotlib.pyplot as plt
import numpy as np


def plot_mean_variance(dist_name, array, mu, sigma):
    # plt
    plt.subplot(121)  # subplot(rows, cols, # of subplot)
    plt.title("Plot")
    plt.xlabel("X(m)")
    plt.ylabel("m")
    plt.plot(array)
    # Histogram
    plt.subplot(122)
    plt.title("Histogram")
    plt.xlabel("m")
    plt.ylabel("X(m)")
    count, bins, ignored = plt.hist(array, bins='auto', density=True)
    plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(bins - mu) ** 2 / (2 * sigma ** 2)), linewidth=2,
             color='r')
    plt.axvline(x=mu, ymax=1, color='orange', label='Expected value is {}'.format(mu))
    # Name and plot
    plt.legend()
    plt.suptitle(dist_name)
    plt.show()


# The (Weak) Law of Large Numbers (LLN) says that the distribution of X^*_n
# concentrates around its average for large n:
def plot_mean(dist_name, array, mu):
    # plt
    plt.subplot(121)  # subplot(rows, cols, # of subplot)
    plt.title("Plot")
    plt.xlabel("X(m)")
    plt.ylabel("m")
    plt.plot(array)
    # Histogram
    plt.subplot(122)
    plt.title("Histogram")
    plt.xlabel("m")
    plt.ylabel("X(m)")
    plt.hist(array, bins='auto', density=True)
    plt.axvline(x=mu, ymax=1, color='orange', label='Expected value is {}'.format(mu))
    # Name and plot
    plt.legend()
    plt.suptitle(dist_name)
    plt.show()


# The Central Limit Theorem states the the shape on the limit does not depend on the shape of the initial
# distribution. It will always be Gaussian.
def plot_variance(dist_name, array, mu, sigma):
    # plt
    plt.subplot(121)  # subplot(rows, cols, # of subplot)
    plt.title("Plot")
    plt.xlabel("X(m)")
    plt.ylabel("m")
    plt.plot(array)
    # Histogram
    plt.subplot(122)
    plt.title("Histogram")
    plt.xlabel("m")
    plt.ylabel("X(m)")
    count, bins, ignored = plt.hist(array, bins='auto', density=True)
    plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(bins - mu) ** 2 / (2 * sigma ** 2)), linewidth=2,
             color='r')
    # Name and plot
    plt.suptitle(dist_name)
    plt.show()


# Docs: https://matpltlib.org/stable/api/_as_gen/matpltlib.pyplt.quiver.html
def plot_multivariate_gaussian(x, y, mean1, dir11, dir12, mean2, dir21, dir22):
    # TODO: find a non hard-coded scale
    plt.title("Multivariate gaussian")
    plt.axis('equal')
    plt.scatter(x, y)
    plt.quiver(*mean1, dir11, dir12, angles='xy', scale_units='xy', scale=0.5, color='red')
    plt.quiver(*mean2, dir21, dir22, angles='xy', scale_units='xy', scale=0.5, color='red')
    theta = np.linspace(0, 2 * np.pi, 1000)
    a1 = [dir11, dir12]
    a2 = [dir21, dir22]
    ellipsis = np.matmul([a1, a2], [np.sin(theta), np.cos(theta)])
    plt.plot(ellipsis[0, :], ellipsis[1, :])
    plt.show()


def plot_circle_A(x, y, mean1, dir11, dir12, mean2, dir21, dir22):
    plt.title("6 experiment Circle ")
    plt.axis('equal')
    plt.scatter(x, y)
    plt.show()

def plot_elipse_A(x, y, mean1, dir11, dir12, mean2, dir21, dir22):
    plt.title("6 experiment Ellipse Matrix A")
    plt.axis('equal')
    plt.plot(x, y)
    plt.quiver(*mean1, dir11, dir12, angles='xy', scale_units='xy', scale=2, color='red')
    plt.quiver(*mean2, dir21, dir22, angles='xy', scale_units='xy', scale=2, color='blue')
    plt.show()


def plot_elipse_B(x, y, mean1, dir11, dir12, mean2, dir21, dir22):
    plt.title("6 experiment Ellipse matrix B")
    plt.axis('equal')
    plt.plot(x, y)
    plt.quiver(*mean1, dir11, dir12, angles='xy', scale_units='xy', scale=1, color='red')
    plt.quiver(*mean2, dir21, dir22, angles='xy', scale_units='xy', scale=1, color='blue')
    plt.show()


def plot_colA_KerAt(colA,kerAt):
    plt.title("6 experiment Col A Ker At")
    plt.axis('equal')
    plt.axvline(x=0, ymax=1, color='black')
    plt.axhline(y=0, xmax=1, color='black')
    plt.quiver(0, 0, colA[0], colA[1], angles='xy', scale_units='xy', scale=50, color='red')
    plt.quiver(0, 0, kerAt[0], kerAt[1], angles='xy', scale_units='xy', scale=50, color='blue')
    plt.show()


def plot_colAt_KerA(colAt,kerA):
    plt.title("6 experiment Col At Ker A")
    plt.axis('equal')
    plt.axvline(x=0, ymax=1, color='black')
    plt.axhline(y=0, xmax=1, color='black')
    plt.quiver(0, 0, colAt[0], colAt[1], angles='xy', scale_units='xy', scale=50, color='red')
    plt.quiver(0, 0, kerA[0], kerA[1], angles='xy', scale_units='xy', scale=50, color='blue')
    plt.show()

def plot_colB_KerBt(colB,kerBt):
    plt.title("6 experiment Col B Ker Bt")
    plt.axis('equal')
    plt.axvline(x=0, ymax=1, color='black')
    plt.axhline(y=0, xmax=1, color='black')
    plt.quiver(0, 0, colB[0][0], colB[0][1], angles='xy', scale_units='xy', scale=50, color='red')
    plt.quiver(0, 0, colB[1][0], colB[1][1], angles='xy', scale_units='xy', scale=50, color='blue')
    plt.show()


def plot_colBt_KerB(colBt,kerB):
    plt.title("6 experiment Col Bt Ker B")
    plt.axis('equal')
    plt.axvline(x=0, ymax=1, color='black')
    plt.axhline(y=0, xmax=1, color='black')
    plt.quiver(0, 0, colBt[0][0], colBt[0][1], angles='xy', scale_units='xy', scale=50, color='red')
    plt.quiver(0, 0, colBt[1][0], colBt[1][1], angles='xy', scale_units='xy', scale=50, color='blue')
    plt.show()

def plot_colC_KerCt(colC,kerCt):
    plt.title("6 experiment Col C Ker Ct")
    plt.axis('equal')
    plt.axvline(x=0, ymax=1, color='black')
    plt.axhline(y=0, xmax=1, color='black')
    plt.quiver(0, 0, colC, colC, angles='xy', scale_units='xy', scale=50, color='red')
    plt.quiver(0, 0, kerCt, kerCt, angles='xy', scale_units='xy', scale=50, color='blue')
    plt.show()


def plot_colCt_KerC(colCt,kerC):
    plt.title("6 experiment Col Ct Ker C")
    plt.axis('equal')
    plt.axvline(x=0, ymax=1, color='black')
    plt.axhline(y=0, xmax=1, color='black')
    plt.quiver(0, 0, colCt[0], colCt[1], angles='xy', scale_units='xy', scale=50, color='red')
    plt.quiver(0, 0, kerC[0], kerC[1], angles='xy', scale_units='xy', scale=50, color='blue')
    plt.show()

def plot_colD_KerDt(colD,kerDt):
    plt.title("6 experiment Col D Ker Dt")
    plt.axis('equal')
    plt.axvline(x=0, ymax=1, color='black')
    plt.axhline(y=0, xmax=1, color='black')
    plt.quiver(0, 0, colD[0], colD[1], angles='xy', scale_units='xy', scale=50, color='red')
    plt.quiver(0, 0, kerDt[0], kerDt[1], angles='xy', scale_units='xy', scale=50, color='blue')
    plt.show()


def plot_colDt_KerD(colDt,kerD):
    plt.title("6 experiment Col Dt Ker D")
    plt.axis('equal')
    plt.axvline(x=0, ymax=1, color='black')
    plt.axhline(y=0, xmax=1, color='black')
    plt.quiver(0, 0, colDt, colDt, angles='xy', scale_units='xy', scale=50, color='red')
    plt.quiver(0, 0, kerD, kerD, angles='xy', scale_units='xy', scale=50, color='blue')
    plt.show()


def plot_arrows_circle(circle_x, circle_y, radius, xlim, ylim, mean1, dir11, dir12, mean2, dir21, dir22):
    circle = plt.Circle((circle_x, circle_y), radius, fill=False, color='blue')
    plt.gca().add_patch(circle)
    plt.axis('equal')
    plt.xlim([-xlim, xlim])
    plt.ylim([-ylim, ylim])
    plt.axvline(x=0, ymax=1, color='black')
    plt.axhline(y=0, xmax=1, color='black')
    plt.quiver(*mean1, dir11, dir12, angles='xy', scale_units='xy', scale=1, color='red')
    plt.quiver(*mean2, dir21, dir22, angles='xy', scale_units='xy', scale=1, color='red')
    plt.show()


def plot_arrows_ellipse(xlim, ylim, mean1, dir11, dir12, mean2, dir21, dir22):
    plt.axis('equal')
    plt.xlim([-xlim, xlim])
    plt.ylim([-ylim, ylim])
    plt.axvline(x=0, ymax=1, color='black')
    plt.axhline(y=0, xmax=1, color='black')
    plt.quiver(*mean1, dir11, dir12, angles='xy', scale_units='xy', scale=1, color='red')
    plt.quiver(*mean2, dir21, dir22, angles='xy', scale_units='xy', scale=1, color='red')
    theta = np.linspace(0, 2 * np.pi, 1000)
    a1 = [dir11, dir12]
    a2 = [dir21, dir22]
    ellipsis = np.matmul([a1, a2], [np.sin(theta), np.cos(theta)])
    plt.plot(ellipsis[0, :], ellipsis[1, :])
    plt.show()
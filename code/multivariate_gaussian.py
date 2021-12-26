import numpy as np

M = 10000
mean = [3, 3]  # µ = (3, 3)
sigma = [[5 / 2, -1 / 2], [-1 / 2, 5 / 2]]  # Σ => covariance matrix


# Generate samples
def samples():
    rng = np.random.default_rng()
    norm = rng.multivariate_normal(mean, sigma, size=M)

    # norm is a 2D array of with M elements [x,y]
    # we need two 1D arrays with x's and y's separated in order to be able to do scatter plot
    x = []
    y = []
    for xn, yn in norm:
        x.append(xn)
        y.append(yn)

    return x, y


# Generate (x1, x2)
def arrows():
    # Generate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(sigma)

    # Each arrowX will be [[mean], dir1, dir2]
    # being mean the coordinates of the arrow locations and dirX the direction components
    # a1 is first eigenvalue multiplied by xn of the eigenvectors
    # a2 is second eigenvalue multiplied by yn of the eigenvectors
    arrow1 = [mean]
    arrow2 = [mean]
    for x1, x2 in eigenvectors:
        a1 = x1 * eigenvalues[0]
        a2 = x2 * eigenvalues[1]
        arrow1.append(a1)
        arrow2.append(a2)
    return arrow1, arrow2

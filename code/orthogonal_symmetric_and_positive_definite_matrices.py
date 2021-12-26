import numpy as np
import math


A = np.array([[math.sqrt(2) / 2, math.sqrt(2) / 2], [math.sqrt(2) / 2, -math.sqrt(2) / 2]])   # Rotation matrix
B = np.array([[3, 5], [5, 2]])
C = np.array([[3, 1], [1, 2]])    # both eigenvalues are positive


def eigen_a():
    eigenvalues_a, eigenvectors_a = np.linalg.eig(A)
    return eigenvalues_a, eigenvectors_a


def eigen_b():
    eigenvalues_b, eigenvectors_b = np.linalg.eig(B)
    return eigenvalues_b, eigenvectors_b


def eigen_c():
    eigenvalues_c, eigenvectors_c = np.linalg.eig(C)
    return eigenvalues_c, eigenvectors_c


# Almost all vectors change direction when they are multiplied by matrix A. Certain exceptional vectors x are in the
# same direction as Ax. Those are the “eigenvectors”. All other vectors are combinations of the two eigenvectors
# Ax = λx  ->  λ is the eigenvalue of A
# Try to prove det(A - λI) = 0
def arrows_a():
    eigenvalues_a, eigenvectors_a = eigen_a()
    arrow1 = [[0, 0]]
    arrow2 = [[0, 0]]
    for x1, x2 in eigenvectors_a:
        a1 = x1 * eigenvalues_a[0]
        a2 = x2 * eigenvalues_a[1]
        arrow1.append(a1)
        arrow2.append(a2)
    return arrow1, arrow2


def arrows_b():
    eigenvalues_b, eigenvectors_b = eigen_b()
    arrow1 = [[0, 0]]
    arrow2 = [[0, 0]]
    for x1, x2 in eigenvectors_b:
        a1 = x1 * eigenvalues_b[0]
        a2 = x2 * eigenvalues_b[1]
        arrow1.append(a1)
        arrow2.append(a2)
    return arrow1, arrow2


def arrows_c():
    eigenvalues_c, eigenvectors_c = eigen_c()
    arrow1 = [[0, 0]]
    arrow2 = [[0, 0]]
    for x1, x2 in eigenvectors_c:
        a1 = x1 * eigenvalues_c[0]
        a2 = x2 * eigenvalues_c[1]
        arrow1.append(a1)
        arrow2.append(a2)
    return arrow1, arrow2

"""
# S = QAQ^t
qa = np.array([[eigenvectors_a[0][0], eigenvectors_a[1][0]], [eigenvectors_a[0][1], eigenvectors_a[1][1]]])
qd = np.array([[eigenvalues_a[0], 0], [0, eigenvalues_a[1]]])
qa_t = np.transpose(qa)

qa_qd = np.matmul(qa, qd)
s_a = np.matmul(qa_qd, qa_t)
print(s_a)
"""
import numpy as np
import math

mean = [0, 0]
A=np.array([[2 , 2], [3 , 3]])
B= np.array([[2 , 1], [1 , 3]])
C= np.array([[2 , -2]])
D= np.array([[ 2 ], [ 3 ]])

def circle():
    x = []
    y = []
    for i in range(0,49):
        x.append(math.cos((i*2*math.pi)/50))
        y.append(math.sin((i*2*math.pi)/50))
    return x, y

def elipse_A():
    x = []
    y = []
    xf = []
    yf = []
    for i in range(0,49):
        x.append(math.cos((i*2*math.pi)/50))
        y.append(math.sin((i*2*math.pi)/50))

    for i in range(0,49):
        x1, y1= trasformation_A(x[i], y[i])
        xf.append(x1)
        yf.append(y1)
    return xf, yf

def elipse_B():
    x = []
    y = []
    xf = []
    yf = []
    for i in range(0,49):
        x.append(math.cos((i*2*math.pi)/50))
        y.append(math.sin((i*2*math.pi)/50))

    for i in range(0,49):
        x1, y1= trasformation_B(x[i], y[i])
        xf.append(x1)
        yf.append(y1)
    return xf, yf

# Generate (x1, x2)
def arrows_A():
    # Generate eigenvalues and eigenvectors
    eigenvalue_A, eigenvectors_A= np.linalg.eig(A)


    # Each arrowX will be [[mean], dir1, dir2]
    # being mean the coordinates of the arrow locations and dirX the direction components
    # a1 is first eigenvalue multiplied by xn of the eigenvectors
    # a2 is second eigenvalue multiplied by yn of the eigenvectors
    arrow1 = [mean]
    arrow2 = [mean]
    for x1, x2 in eigenvectors_A:
        a1 = x1 * eigenvalue_A[0]
        a2 = x2 * eigenvalue_A[1]
        arrow1.append(a1)
        arrow2.append(a2)
    return arrow1, arrow2

def arrows_B():
    # Generate eigenvalues and eigenvectors
    eigenvalue_B, eigenvectors_B= np.linalg.eig(B)


    # Each arrowX will be [[mean], dir1, dir2]
    # being mean the coordinates of the arrow locations and dirX the direction components
    # a1 is first eigenvalue multiplied by xn of the eigenvectors
    # a2 is second eigenvalue multiplied by yn of the eigenvectors
    arrow1 = [mean]
    arrow2 = [mean]
    for x1, x2 in eigenvectors_B:
        a1 = x1 * eigenvalue_B[0]
        a2 = x2 * eigenvalue_B[1]
        arrow1.append(a1)
        arrow2.append(a2)
    return arrow1, arrow2

def trasformation_A(x,y):
    return 2*x + 2*y, 3*x + 3*y

def trasformation_B(x,y):
    return 2*x + 1*y, 1*x + 3*y


def svdA():
    uA,sA,vA= np.linalg.svd(A)
    rankA=np.linalg.matrix_rank(A)
    uAt,sAt,vAt= np.linalg.svd(A.transpose())
    col_spaceA=uA[0][:]
    null_spaceA=vA[:][1]
    col_spaceAt=uAt[0][:]
    null_spaceAt=vAt[:][1]
    return col_spaceA, null_spaceA, col_spaceAt, null_spaceAt

def svdB():
    uB,sB,vB= np.linalg.svd(B)
    rankB=np.linalg.matrix_rank(B)
    uBt,sBt,vBt= np.linalg.svd(B.transpose())
    col_spaceB=(uB[0][:], uB[1][:])
    null_spaceB=0
    col_spaceBt=(uBt[0][:], uBt[1][:])
    null_spaceBt=0
    return col_spaceB, null_spaceB, col_spaceBt, null_spaceBt

def svdC():
    uC,sC,vC= np.linalg.svd(C)
    rankC=np.linalg.matrix_rank(C)
    uCt,sCt,vCt= np.linalg.svd(C.transpose())
    col_spaceC=uC[0][0]
    null_spaceC=vC[1][0], vC[1][1]
    col_spaceCt=uCt[0][0], uCt[1][0]
    null_spaceCt=0
    return col_spaceC, null_spaceC, col_spaceCt, null_spaceCt


def svdD():
    uD,sD,vD= np.linalg.svd(D)
    rankD=np.linalg.matrix_rank(D)
    uDt,sDt,vDt= np.linalg.svd(D.transpose())
    col_spaceD=uD[0][0], uD[1][0]
    null_spaceD=0
    col_spaceDt=uDt[0][0]
    null_spaceDt=vDt[1][0],vDt[1][1]
    return col_spaceD, null_spaceD, col_spaceDt, null_spaceDt

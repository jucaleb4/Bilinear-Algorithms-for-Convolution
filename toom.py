import numpy as np
import numpy.linalg as la
from math import log,ceil,floor

# produces smallest magn. ints: [1,-1,2,-2,3,...]
def rs(k):
    return np.asarray([
        (-1)**(i+1) * ceil(i/2) for i in range(k)
    ])

def Q_mat(gamma,eta):
    n = gamma * eta
    w = (2*gamma-1)*(2*eta-1)
    Q = np.zeros((2*n-1, w))

    for i in range(2*n-1):
        for j in range(w):
            if i == j - (eta-1)*floor(j/(2*eta-1)):
                Q[i,j] = 1 
    return Q

def toom_cook_mats_w_pts(r, n, pts):
    assert(len(pts) == r+n-1)
    use_infty = np.infty in pts
    if(use_infty and pts[-1] != np.infty):
        raise Exception('np.infty must be the last node choice')

    V = np.vander(pts,increasing=True)
    if(use_infty):
        V[-1,:] = 0; V[-1,-1] = 1

    A = V[:,:r].copy();
    B = V[:,:n].copy();
    if(use_infty):
        A[-1,-1] = 1
        B[-1,-1] = 1

    C = la.inv(V)
    return [A.T,B.T,C]

def toom_cook_mats(r, n, cheby=False):
    pts = rs(n+r-1)
    if cheby:
        z = r+n-1
        i = np.arange(z, dtype=np.float64)
        pts = np.cos((2*i+1)/(2*z)*np.pi)

    V = np.vander(pts,increasing=True)
    V[-1,:] = 0; V[-1,-1] = 1
    C = la.inv(V)

    A = V[:,:r].copy(); A[-1,-1] = 1
    B = V[:,:n].copy(); B[-1,-1] = 1

    return [A.T,B.T,C]

def prime_list(x):
    if x == 1: return np.asarray([1])
    for f in range(2,x):
        if x % f == 0:
            return np.append(f,prime_list(x//f))
    return np.asarray([x])

def auto_nested_toom_cook(n):
    prime_nums = prime_list(n)
    return nested_toom_cook(prime_nums)

# list_k is list of decompositions we want
def nested_toom_cook(list_k):
    if(len(list_k) == 0):
        return [np.eye(1),np.eye(1),np.eye(1)]

    post_k = 1
    for k in list_k[1:]:
        post_k *= k

    k = list_k[0]
    [A1,B1,C1] = toom_cook_mats(k,k)
    [A2,B2,C2] = nested_toom_cook(list_k[1:])
    Q = Q_mat(k,post_k)

    return [np.kron(A1,A2), np.kron(B1,B2), np.dot(Q, np.kron(C1,C2)) ]

###############################
### Discrete Fourier Matrix ###
###############################
## For FFT
def omega(n):
    return np.exp(-2*np.pi*1j/n)

def F_matrix(n, N=-1, offset=0):
    w = omega(n if N==-1 else N)
    F = np.zeros((n,n), dtype=complex)
    for i in range(n):
        for j in range(n):
            F[i,j] = w**(i*(j+offset))
    return F

def F_matrix_inverse(n):
    w = omega(n)
    Finv = np.zeros((n,n), dtype=complex)
    for i in range(n):
        for j in range(i,n):
            Finv[i,j] = Finv[j,i] = w**(-1*i*j)/n
    return Finv

def dft_matrices(r, n, cyclic=False):
    assert(r <= n)
    z = (n if cyclic else n+r-1)
    F = F_matrix(z)
    Finv = F_matrix_inverse(z)

    return [F[:,:r].copy().T,F[:,:n].copy().T,Finv]

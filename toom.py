import numpy as np
import numpy.linalg as la
from math import log,ceil,floor

# produces smallest magn. ints: [1,-1,2,-2,3,...]
def rs(k):
    return np.asarray([
        (-1)**(i+1) * ceil(i/2) for i in range(k)
    ])

def Q_mat(s,n):
    assert(s >= n and s%n==0)
    z = s//n # smaller size
    w = 2*z-1 # smaller conv size
    Q = np.zeros((2*s-1, (2*n-1)*w))

    for i in range(2*n-1):
        row = i*z
        col = i*w
        Q[row:row+w, col:col+w] += np.eye(w)

    return Q

def toom_cook_mats_w_pts(r, n, pts):
    assert(len(pts) == r+n-1)
    use_infty = np.infty in pts
    if(use_infty and pts[-1] != np.infty):
        print("np.infty must be the last node choice")
        assert(False)

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
    assert(len(list_k ) > 0)
    n = 1
    for k in list_k:
        n*=k

    start_k = list_k[0]

    [A,B,C] = toom_cook_mats(start_k,start_k)
    AA = A.copy()
    BB = B.copy()
    CC = np.kron( np.eye(1) ,
                    np.dot(Q_mat(n, start_k),
                        np.kron(C, np.eye(2*n//start_k-1) )
                    )
                )

    pre_total = start_k
    post_total = 2*start_k-1
    for k in list_k[1:]:
        k = int(k)
        [A,B,C] = toom_cook_mats(k,k)

        AA = np.kron(A,AA)
        BB = np.kron(B,BB)

        Q = Q_mat(n//pre_total,k)
        CC = np.dot(CC, np.kron( np.eye(post_total) ,
                        np.dot(Q, np.kron(C, np.eye(2*n//(pre_total*k)-1) )
                            )
                        )
                    )
        pre_total *= k
        post_total *= 2*k-1

    return [AA,BB,CC]

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

    return [F[:,:r].copy().T,F[:,:b].copy().T,Finv]

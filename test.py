import numpy as np
import numpy.linalg as la
from wino import *
from toom import *

def idx_replace(s,c,i):
    sList = list(s)
    assert(len(sList) >= i+1 >= 0)
    sList[i] = c
    return ''.join(sList)

def idx_replace(s,c,i):
    sList = list(s)
    assert(len(sList) >= i+1 >= 0)
    sList[i] = c
    return ''.join(sList)

def contract_nD(A,M):
    nD = len(M.shape)
    assert(nD <= 24)
    idx = ""
    start = ord('z') - (nD-1)

    for i in range(nD-1):
        idx += str(chr(start + i))
    idx += "z"

    K = M.copy()
    for i in range(nD):
        temp = idx_replace(idx,'K',i)
        temp2 = idx_replace(idx,'I',i)
        msg = "IK," + temp + "->" + temp2
        K = np.einsum(msg,A,K)
    return K

def compute_bilinear_algorithm(bi_alg,F,G):
    [A,B,C] = bi_alg
    F2 = contract_nD(A.T,F)
    G2 = contract_nD(B.T,G)
    Y = contract_nD(C,F2*G2)
    return Y

def direct_conv(F,G):
    nn = G.shape; rr = F.shape
    assert(len(nn) == len(rr))

    is1D = len(nn) == 1

    n = nn[0]; r = rr[0]
    R = n + r - 1
    Y = np.zeros((R,)*len(nn))

    for k in range(R):
        for i in range(r):
            if(is1D):
                Y[k] += 0 if k-i<0 or k-i>=n else F[i]*G[k-i]
            else:
                if(0 <= k-i < n):
                    Y[k] += direct_conv(F[i],G[k-i])
    return Y

def relative_error(Y_act, Y_comp):
    Y_diff = Y_act - Y_comp
    return la.norm(Y_diff)/la.norm(Y_act)

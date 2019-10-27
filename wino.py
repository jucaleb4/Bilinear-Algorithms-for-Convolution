import numpy as np
import numpy.linalg as la
import toom

# Polynomial Object that stores the coefficients in a vector
class Polynomial:
    def __init__(self, coefficients):
        self.coeffs = coefficients

        # may have zero leading terms
        max_idx = -1 
        for i in range(len(coefficients)):
            if(coefficients[i] != 0):
                max_idx = i
        assert(max_idx >= 0)
        self.degree = i

    def get_coeffs(self):
        return self.coeffs

    def get_degree(self):
        return self.degree

def poly_prod(p1,p2):
    prod_poly = np.convolve(p1.get_coeffs(), p2.get_coeffs())
    return Polynomial(prod_poly)

def get_all_poly_prod(p_list):
    poly = p_list[0]
    for i in range(1,len(p_list)):
        poly = poly_prod(poly, p_list[i])
    return poly

def get_all_poly_prod_except_i(p_list,i):
    p_list2 = np.asarray([])
    for j in range(len(p_list)):
        if(j != i):
            p_list2 = np.append(p_list2, p_list[j])
    return get_all_poly_prod(p_list2)

def T_conv(poly,input_size):
    p_coeffs = poly.get_coeffs()
    p_num_coeffs = len(p_coeffs)
    H = np.zeros((p_num_coeffs + input_size-1, input_size))
    for col_idx in range(input_size):
        H[col_idx:col_idx + p_num_coeffs, col_idx] = p_coeffs
    return H

def X_eval(m,input_deg):
    deg_m = m.get_degree()

    # trivial modulo
    if(deg_m > input_deg):
        X = np.zeros((deg_m,input_deg+1))
        X[:input_deg+1,:input_deg+1] = np.eye(input_deg+1)
        return X

    T = T_conv(m,input_deg-deg_m+1)
    L = T[:deg_m,:]; U = T[-(input_deg-deg_m+1):,:]
    I = np.eye(deg_m)
    return np.hstack([I, np.dot(-L,la.inv(U)) ])

def Ni_coeffs(M,m):
    T_M = T_conv(M, m.get_degree())
    T_m = T_conv(m, M.get_degree())
    A = np.hstack([T_M,T_m])
    b = np.zeros(A.shape[0]); b[0] = 1
    N_coeffs = la.solve(A,b)[:T_M.shape[1]]
    return Polynomial(N_coeffs)

def winograd_conv_mats(polys,r,n):
    M = get_all_poly_prod(polys)
    deg_M = M.get_degree()

    A = None
    B = None
    C = None
    first = True

    for i,mi in enumerate(polys):
        deg_m = mi.get_degree()
        [A_i,B_i,C_i] = toom.toom_cook_mats(deg_m,deg_m)

        # create matrix A
        X = X_eval(mi,r-1)
        A_i = np.dot(X.T,A_i)
        A = (A_i if first else np.hstack([A,A_i]))

        # create matrix B
        X = X_eval(mi,n-1)
        B_i = np.dot(X.T,B_i)
        B = (B_i if first else np.hstack([B,B_i]))

        # compute e_i = M_iN_i (no mod)
        Mi = get_all_poly_prod_except_i(polys,i)
        Ni = Ni_coeffs(Mi, mi)
        e_i = poly_prod(Mi,Ni)

        # compute e_i = M_iN_i mod M
        T_i = X_eval(M,e_i.get_degree())
        e_i_coeffs = np.dot(T_i,e_i.get_coeffs())
        e_i_filler = np.zeros(deg_M)
        e_i_filler[:len(e_i_coeffs)] = e_i_coeffs
        e_i = Polynomial(e_i_coeffs)

        # bring it all together
        T_ei = T_conv(e_i, deg_m)
        X_M = X_eval(M, deg_M + deg_m - 2)

        # create matrix C
        X_m = X_eval(mi,2*deg_m-2)
        C_i = np.dot( np.dot(X_M, T_ei), np.dot(X_m,C_i) )
        C = (C_i if first else np.hstack([C,C_i]))
        first = False

    return [A,B,C]

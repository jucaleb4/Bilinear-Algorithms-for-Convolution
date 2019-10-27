import numpy as np
import wino
from sympy.polys import ring, QQ

RR, x = ring("x", QQ)

def sym_deg(p):
    return p.terms()[0][0][0]

def sym_coeffs(p):
    # extracts all coefficients form 1 -> x**p
    d = sym_deg(p)
    terms = p.to_dict()
    coeffs = np.zeros(d+1, dtype=complex)
    for i in range(d+1):
        v = tuple([i])
        if v in terms:
            coeffs[i] = terms[v]
    return coeffs

def winograd_mats_symbol(polys,r,n):
    polynomial_list = np.asarray([])
    for p in polys:
        p_coeffs = sym_coeffs(p)
        P = wino.Polynomial(p_coeffs)
        polynomial_list = np.append(polynomial_list, P)
    return wino.winograd_conv_mats(polynomial_list, r, n)

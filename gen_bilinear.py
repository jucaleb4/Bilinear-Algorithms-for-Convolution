import numpy as np
import wino
import toom
import test

r = 2 # filter size
n = 4 # input size

#####################################
### Toom-Cook w/ generated points ###
#####################################
# [A,B,C] = toom.toom_cook_mats(r,n, cheby=False)


###############################
### Toom-Cook w/ own points ###
###############################
# pts = np.asarray([0,1,,np.infty])
# [A,B,C] = toom.toom_cook_mats_w_pts(r,n,pts)


#######################################################
### Nested Toom-Cook (n=r only) with generic decomp ###
#######################################################
# [A,B,C] = toom.auto_nested_toom_cook(n)


########################################################
### Nested Toom-Cook (n=r only) with supplied decomp ###
########################################################
# decomp_list = np.asarray([2,...])
# [A,B,C] = toom.nested_toom_cook(decomp_list)


#######################################
### Winograd w/ vector coefficients ###
#######################################
# Ex) Generates m_i: x**2+1,x,x-1,x+1
# polys = np.asarray([
#    wino.Polynomial(np.asarray([2,0,1])),
#    wino.Polynomial(np.asarray([0,1])),
#    wino.Polynomial(np.asarray([1,1])),
#    wino.Polynomial(np.asarray([-1,1]))
# ])
# [A,B,C] = wino.winograd_conv_mats(polys,r,n)


#############################################
### Winograd w/ Sympy polynomial notation ###
#############################################
# import symbols
# from sympy.polys import ring, QQ
# RR, x = ring("x", QQ)

# polys = np.asarray([x**2+2,x,x-1,x+1])
# [A,B,C] = symbols.winograd_mats_symbol(polys,r,n)

################################
### test with 1D convolution ###
################################
# f = np.random.random(r)
# g = np.random.random(n)
# y_comp = test.compute_bilinear_algorithm([A,B,C],f,g)
# y_direct = test.direct_conv(f,g)
# rel_err = test.relative_error(y_direct,y_comp)


######################################
### test with multidim convolution ###
######################################
# dim = 3
# F = np.random.random((r,) * dim)
# G = np.random.random((n,) * dim)
# Y_comp = test.compute_bilinear_algorithm([A,B,C],F,G)
# Y_direct = test.direct_conv(F,G)
# rel_err = test.relative_error(Y_direct,Y_comp)

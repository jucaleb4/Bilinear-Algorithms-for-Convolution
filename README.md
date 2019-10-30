# Generating Fast Bilinear Algorithms for different convolution algorithms

A Python module to generate fast bilinear algorithms for different variants of
convolution.

Requirements
+ python: version 3.7.0
+ numpy: version 1.17.3

Optional
+ sympy: version 1.4

## Bilinear algorithms for linear convolution
We describe a class of fast convolution algorithms using the matrices `[A,B,C]`.

To generate these matrices, we provide a variety of methods, which
can be called from the `gen_bilinear.py` file. Let `r` be the
filter size and `n` be the input size.

### Toom-Cook
We provide a simple function to generate Toom-Cook algorithms
with either integer nodes `(cheby=False)` or Chebyshev nodes
`(cheby=True)`.
```
r = 2
n = 3
[A,B,C] = toom.toom_cook_mats(r,n, cheby=False)
```

Alternatively, the nodes can be prescribed beforehand. To use
the infinity node point, simply designate the last node to
be `np.infty`. The user must ensure the number of points is 
equal to `n+r-1`.
```
r = 3
n = 3
pts = np.asarray([0,1,-1,2,np.infty])
[A,B,C] = toom.toom_cook_mats_w_pts(r,n,pts)
```

### Nested Toom-Cook
We provide a function to generate nested Toom-Cook algorithms.
Based on the size `n`, our algorithm decomposes it to its
prime values e.g. the value of `n=8` is set to a `2x2x2` 
nesting.
```
n = 8
[A,B,C] = toom.auto_nested_toom_cook(n)
```

The user can also define the decomposition him or herself.
For example, to specify a `2x4` decomposition, the user 
supplies the following list,
```
decomp_list = np.asarray([2,4])
[A,B,C] = toom.nested_toom_cook(decomp_list)
```

### Winograd convolution algorithm
We provide a function to generate Winograd's convolution algorithm.
Unlike previous methods, the user must define their own polynomial
divisors. These polynomials will be described using the
`Polynomial` object in the `wino.py` file. It takes in a
vector of coefficients in increasing order. For example,
the polynomial `x**2 - 1` is described by the vector of coefficients,
`[-1,0,1]`. Below is an example for an algorithm described by
the divisors, `x**2+1,x,x-1,x+1`.

```
polys = np.asarray([
   wino.Polynomial(np.asarray([1,0,1])),
   wino.Polynomial(np.asarray([0,1])),
   wino.Polynomial(np.asarray([1,1])),
   wino.Polynomial(np.asarray([-1,1]))
])
[A,B,C] = wino.winograd_conv_mats(polys,r,n)
```

To avoid the verbose notation of the `Polynomial` object, we
have also supplied a function to convert polynomials written
using Sympy notation. This function requires the Sympy 
module.
```
import symbols
from sympy.polys import ring, QQ
RR, x = ring("x", QQ)

polys = np.asarray([x**2+1,x,x-1,x+1])
[A,B,C] = symbols.winograd_mats_symbol(polys,r,n)
```

## Computing the bilinear algorithm
Given a bilinear algorithm, `[A,B,C]`, filter `F` and input `G`, the linear
convolution for `F * G` can be computed by calling
`compute_bilinear_algorithm([A,B,C],F,G)` from the `test.py` file. We also
supplied a function `direct_conv(F,G)` to compute the linear convolution for
any inputs of the same dimension.

## Generating correlation convolution algorithms
The Matrix Interchange shows that the bilinear algorithm for 
correlation algorithms is `[A,C,B]`, where the bilinear algorithm
for linear convolution is `[A,B,C]`.

## Generating circular convolution algorithms
Consider the cyclic convolution of two inputs of size `n`. It suffices
to compute a Winograd convolution algorithm defined by the product
modulo space `M=x**n-1` to generate a bilinear algorithm for a
`n`-cyclic convolution algorithm. Therefore, generating a Winograd's
convolution algorithm with modulus that are the factors of `x**n-1`
will create a sufficient bilinear algorithm.

Below is an example for generating an 8-cyclic convolution algorithm.
```
n = 8
polys = np.asarray([
   wino.Polynomial(np.asarray([1,0,0,0,1])),
   wino.Polynomial(np.asarray([1,0,1])),
   wino.Polynomial(np.asarray([1,1])),
   wino.Polynomial(np.asarray([-1,1]))
])
[A,B,C] = wino.winograd_conv_mats(polys,n,n)
```

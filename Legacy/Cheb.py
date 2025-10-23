import numpy as np
from flint import *

dp = 100 #Set decimal places of precision
ctx.dps = dp

def Cheb(f,n):
    #Calculate the polynomial interpolant at the zeroes of T_n
    #Coefficients returned in the Chebychev basis
    x = [arb.cos((2*x - 1)*arb.pi()/(2*n)) for x in range(1,n+1)]
    points = [f(a) for a in x]
    points = points + [arb(0)]*n

    b = acb.dft(points,inverse=True)
    b = [4*acb.exp(1j*(acb.pi()/2/n)*i)*b[i] for i in range(0,n)]
    b[0] /= 2
    return [a.real for a in b[:n]]

def to_list(c, prec = 1e-17):
    #Truncate series to FP-64, ignoring values below a threshold
    out = []
    last = 0
    for i in range(len(c)):
        out.append(float(c[i]))
        if abs(c[i]) > prec:
            last = i
    return out[:last + 1]

def clenshaw(c,x):
    #The Clenshaw algorithm for evaluating a polynomial in the Chebyshev basis.
    n = len(c)
    b2 = 0
    b1 = c[n-1]
    
    for r in range(n-2, 0, -1):
        b1, b2 = 2*x*b1 - b2 + c[r], b1
        
    return x*b1 -b2 + c[0]

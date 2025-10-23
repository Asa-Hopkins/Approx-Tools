from flint import *
import numpy as np

dp = 100 #Set decimal places of precision
ctx.dps = dp

def CF(fz, m, n, nfft, K, real = True):
# CF -- RATIONAL CF APPROXIMATION ON THE UNIT DISC
# Based on the following Matlab code
# Lloyd N. Trefethen, Dept. of Math., M.I.T., March 1986
# Reference: L.N.T. and M.H. Gutknecht,
#            SIAM J. Numer. Anal. 20 (1983), pp. 420-436
#
#    Fz(z) - function to be approximated by R(z)=P(z)/Q(z)
#      m,n - degree of P,Q
#     nfft - number of points in FFT (must be power of 2)
#        K - degree at which Taylor series is truncated
#     real - whether to allow complex coefficients in P and Q
#    pc,qc - Monomial coefficients of P and Q
#        s - estimate of error, more accurate for larger m, n
#
# If Fz is even, take (m,n) = (even,even).
# If Fz is  odd, take (m,n) = (odd,even).
#
# CONTROL PARAMETERS
    nfft2 = nfft//2
    m1 = m
    n1 = n
    dim = K+n-m
#
# TAYLOR COEFFICIENTS OF fz
    z = [acb.exp_pi_i(2 * acb(x) /  nfft) for x in range(0,nfft)]
    f = [fz(x) for x in z]
    fc = [x/nfft for x in acb.dft(f)]
    if real:
        fc = [x.real for x in fc]
    fc[nfft2:] = [0 for x in range(len(fc) - nfft2)]
    
#
# SVD OF HANKEL MATRIX H
    row = [fc[(x + nfft + m - n) % nfft] for x in range(1,dim+1)]
    H = []
    for i in range(0,len(row)):
        H.append(row[i:] + [0]*i)
    H = acb_mat(H)
    s, u = acb_mat.eig(H * H.transpose().conjugate(),left = True, algorithm='approx')
    s, v = acb_mat.eig(H.conjugate().transpose() * H,right = True, algorithm='approx')
    s = [arb.sqrt(x.real) if x.real > 0 else 0 for x in s]
    sort = np.argsort(s)
    s = s[sort[-1-n1]]
    
    u = u.tolist()
    u = u[sort[-1-n1]]#[::-1]

    v = v.transpose().tolist()
    v = v[sort[-1-n1]][::-1]
#
# DENOMINATOR POLYNOMIAL Q
    poly = acb_poly(v)
    zr = acb_poly.roots(poly, tol = 1/(10**dp))
    qout = []
    for i in zr:
        if abs(i) > 1:
            qout.append(i)
    qc = np.poly(qout)
    qc = acb_poly.from_roots(qout)
    if real:
        qc = [x.real for x in qc]
    qc = [x/qc[0] for x in qc]
    
#
# NUMERATOR POLYNOMIAL P
    b = acb.dft(u + [0]*(nfft - dim))
    c = acb.dft(v + [0]*(nfft - dim))
    for i in range(0,len(b)):
        b[i] /= c[i]
    rt = []
    for i in range(0,len(b)):
        rt.append(f[i] - b[i]*s*z[i]**K)
        
    rtc = [x/nfft for x in acb.dft(rt)]
    
    if real:
        rtc = [x.real for x in rtc]
    pc = np.convolve(qc[:n+1], rtc[:m+1])
    pc = pc[:m+1]
#
# RESULTS
    return pc, qc, s

def RCF(fz, m, n, nfft, K, real = True):
# RCF -- RATIONAL CF APPROXIMATION ON THE UNIT INTERVAL
# Based on the following Matlab code
# Lloyd N. Trefethen, Dept. of Math., M.I.T., March 1986
# Reference: L.N.T. and M.H. Gutknecht,
#            SIAM J. Numer. Anal. 20 (1983), pp. 420-436
#
#    Fz(z) - function to be approximated by R(z)=P(z)/Q(z)
#      m,n - degree of P,Q
#     nfft - number of points in FFT (must be power of 2)
#        K - degree at which Taylor series is truncated
#     real - whether to allow complex coefficients in P and Q
#    Pc,Qc - Chebyshev coefficients of P and Q
#        s - estimate of error, more accurate for larger m, n
#
# If Fz is even, take (m,n) = (even,even).
# If Fz is  odd, take (m,n) = (odd,even).
#
# CONTROL PARAMETERS
    nfft2 = nfft//2
    m1 = m
    n1 = n
    dim = K+n-m
#
# TAYLOR COEFFICIENTS OF fz
    z = [acb.exp_pi_i(2 * acb(x) /  nfft) for x in range(0,nfft)]
    y = [x.real for x in z]
    f = [fz(acb(x)) for x in y]
    fc = [x/nfft2 for x in acb.dft(f)]
    if real:
        fc = [x.real for x in fc]
    
#
# SVD OF HANKEL MATRIX H
    row = [fc[(x + nfft + m - n) % nfft] for x in range(1,dim+1)]
    H = []
    for i in range(0,len(row)):
        H.append(row[i:] + [0]*i)
    H = acb_mat(H)
    s, u = acb_mat.eig(H * H.transpose().conjugate(),left = True, algorithm='approx')
    s, v = acb_mat.eig(H.conjugate().transpose() * H,right = True, algorithm='approx')
    s = [arb.sqrt(x.real) if x.real > 0 else 0 for x in s]
    sort = np.argsort(s)
    s = s[sort[-1-n1]]
    
    u = u.tolist()
    u = u[sort[-1-n1]]#[::-1]

    v = v.transpose().tolist()
    v = v[sort[-1-n1]][::-1]
#
# DENOMINATOR POLYNOMIAL Q
    poly = acb_poly(v)
    zr = acb_poly.roots(poly, tol = 1/(10**dp))
    qout = []
    for i in zr:
        if abs(i) > 1:
            qout.append(i)
    qc = np.poly(qout)
    qc = acb_poly.from_roots(qout)
    if real:
        qc = [x.real for x in qc]
    qc = [x/qc[0] for x in qc]
    qc = acb_poly(qc)
    q = []
    q = qc.evaluate(z)
    Q = [x*x.conjugate() for x in q]
    Qc = [x/nfft2 for x in acb.dft(Q)]
    if real:
        Qc = [x.real for x in Qc]
    Qc[0] /= 2
    Q = [x/Qc[0] for x in Q]
    Qc = [x/Qc[0] for x in Qc[:n+1]]
    
#
# NUMERATOR POLYNOMIAL P
    b = acb.dft(v + [0]*(nfft - dim))
    c = acb.dft(u + [0]*(nfft - dim))
    for i in range(0,len(b)):
        b[i] /= c[i]
    #return b
    rt = []
    for i in range(0,len(b)):
        rt.append(f[i] - (b[i]*s*(z[i]**K)).real)
#    return rt

    rtc = [x/nfft2 for x in acb.dft(rt)]
    gam = [x/nfft2 for x in acb.dft([1/y for y in Q])]
    gam = gam[:2*m+1]
    if real:
        rtc = [x.real for x in rtc]
        gam = [x.real for x in gam]
    #return gam
    toep = []
    for i in range(len(gam)):
        if i == 0:
            toep.append(gam)
        else:
            toep.append(gam[:i+1][::-1] + gam[1:-i])
    toep = acb_mat(toep)
    b = acb_mat([[2*x for x in rtc[m:0:-1]+rtc[0:m+1]]])
    if m == 0:
        Pc = rtc[0]/gam[0]
    else:
        Pc = acb_mat.solve(toep,b.transpose())
        Pc = Pc.tolist()
        Pc = Pc[m1:2*m+1]
        Pc = [x[0] for x in Pc]
        Pc[0] /= 2
#
# RESULTS
    return Pc, Qc, s

def to_monomial(x):
    #Convert polynomial in Chebyshev basis to Monomial basis
    return np.polynomial.chebyshev.cheb2poly(x)

def test_error(f,P,Q, real = True):
    if real:
        num = np.polynomial.Chebyshev(P)
        den = np.polynomial.Chebyshev(Q)
    else:
        num = np.polynomial.Polynomial(P)
        den = np.polynomial.Polynomial(Q)
        
    if real:
        a = np.linspace(-1,1,num = 1000)
    else:
        a = np.linspace(0,2,num = 1000)
        a = [acb.exp_pi_i(acb(x)) for x in a]
    err = [abs(f(acb(x)) - num(x)/den(x)) for x in a]
    return max(err)

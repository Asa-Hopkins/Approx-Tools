import numpy as np
import scipy
from mpmath import mp
mp.dps = 50
precision = np.float128
cprecision = np.complex256

def CF(fz, m, n, nfft, K):
# CF -- COMPLEX RATIONAL CF APPROXIMATION ON THE UNIT INTERVAL
#
# Lloyd N. Trefethen, Dept. of Math., M.I.T., March 1986
# Reference: L.N.T. and M.H. Gutknecht,
#            SIAM J. Numer. Anal. 20 (1983), pp. 420-436
#
#    Fx(x) - function to be approximated by R(x)=P(x)/Q(x)
#      m,n - degree of P,Q
#     nfft - number of points in FFT (power of 2)
#        K - degree at which Chebyshev series is truncated
#  F,P,Q,R - functions evaluated on FFT mesh (Chebyshev points)
#    pc,qc - Chebyshev coefficients of P and Q
#
# If Fx is even, take (m,n) = ( odd,even).
# If Fx is  odd, take (m,n) = (even,even).
#
# CONTROL PARAMETERS
    nfft2 = nfft//2
    m1 = m
    n1 = n
    dim = K+n-m
#
# TAYLOR COEFFICIENTS OF fz
    z = np.exp(2*np.pi*1j*np.arange(0,nfft)/nfft)
    f = fz(z)
    fc = np.real(np.fft.fft(f))/nfft
    fc[nfft2:] = np.zeros(len(fc) - nfft2)
#
# SVD OF HANKEL MATRIX H 
    H = scipy.linalg.hankel(fc[(np.arange(1,dim+1) + nfft + m - n) % nfft])
    u, s, v = np.linalg.svd(H)
    s = s[n1]
    u = u[::-1,n1].T
    v = v[n1,:].T
#
# DENOMINATOR POLYNOMIAL Q
    zr = np.roots(v)
    qout = zr[np.abs(zr) > 1]
    qc = np.real(np.poly(qout))
    if type(qc) == type(1.0):
        qc = [1]
    else:
        qc /= qc[n1]
#
# NUMERATOR POLYNOMIAL P
    b = np.fft.fft(np.concatenate((u, np.zeros(nfft - dim))))
    b /= np.fft.fft(np.concatenate((v, np.zeros(nfft - dim))))
    rt = f - b*s*z**K
    rtc = np.real(np.fft.fft(rt))/nfft
    pc = np.convolve(qc[n+1::-1], rtc[:m+1])
    pc = pc[m::-1]
#
# RESULTS
    return pc, qc, s

def RCF(fz, m, n, nfft, K):
# RCF -- REAL RATIONAL CF APPROXIMATION ON THE UNIT INTERVAL
#
# Lloyd N. Trefethen, Dept. of Math., M.I.T., March 1986
# Reference: L.N.T. and M.H. Gutknecht,
#            SIAM J. Numer. Anal. 20 (1983), pp. 420-436
#
#    Fx(x) - function to be approximated by R(x)=P(x)/Q(x)
#      m,n - degree of P,Q
#     nfft - number of points in FFT (power of 2)
#        K - degree at which Chebyshev series is truncated
#  F,P,Q,R - functions evaluated on FFT mesh (Chebyshev points)
#    Pc,Qc - Chebyshev coefficients of P and Q
#
# If Fx is even, take (m,n) = ( odd,even).
# If Fx is  odd, take (m,n) = (even,even).
#
# CONTROL PARAMETERS
    nfft2 = nfft//2
    m1 = m
    n1 = n
    dim = K+n-m
#
# CHEBYSHEV COEFFICIENTS OF fz
    z = np.exp(2*np.pi*1j*np.arange(0,nfft)/nfft)
    x = np.real(z)
    f = fz(x)
    print(f)
    fc = np.real(np.fft.fft(f))/nfft2#
#
# SVD OF HANKEL MATRIX H
    H = scipy.linalg.hankel(fc[(np.arange(1,dim+1) + nfft + m - n) % nfft])
    u, s, v = np.linalg.svd(H)

    s = s[n1]
    u = u[::-1,n1].T
    v = v[n1,:].T
#
# DENOMINATOR POLYNOMIAL Q
    zr = np.roots(v)
    qout = zr[np.abs(zr) > 1]
    qc = np.real(np.poly(qout))
    if type(qc) == type(1.0):
        qc = [1]
    else:
        qc /= qc[n1]
    
    q = np.polyval(qc,z)
    Q = q*np.conj(q)
    Qc = np.real(np.fft.fft(Q))/nfft2
    Qc[0] /= 2
    Q = Q/Qc[0]
    Qc = Qc[:n+1]/Qc[0]
#
# NUMERATOR POLYNOMIAL P
    b = np.fft.fft(np.concatenate((u, np.zeros(nfft - dim))))
    b /= np.fft.fft(np.concatenate((v, np.zeros(nfft - dim))))
    rt = f - np.real(b*s*z**K)
    rtc = np.real(np.fft.fft(rt))/nfft2;
    gam = np.real(np.fft.fft(1/Q)/nfft2)
    gam = scipy.linalg.toeplitz(gam[:2*m+1])
    if m==0:
        Pc = np.linalg.solve(gam,2*rtc[0]);
    else:
        Pc = np.linalg.solve(gam,2*np.concatenate([rtc[m:0:-1],rtc[0:m+1]]));

    Pc = Pc[m1:2*m+1]
    Pc[0] /= 2
#
# RESULTS
    return Pc, Qc, s

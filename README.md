# Approx-Tools

A repository for various useful tools for building polynomial and rational approximations to functions on both complex and real domains. The point of this repository is to make such tools more easily accessible, as many such tools are locked behind the paywall of Matlab like Chebfun (which doesn't work in GNU Octave either), or are entirely proprietary (and also locked behind paywalls) like the Remez function in Maple. The few tools that do exist are often difficult to find, difficult to use, or just generally unfinished.

Currently implemented here is the Carathéodory-Fejér method for creating rational approximations of type (m,n) (allowing n=0 in principle,but output isn't good). These rationals are extremely close to the minimax rationals generated by the Remez algorithm, whilst being a much simpler algorithm to implement. 

Two functions are implemented, CF for approximating a given complex function on the unit disc |z| < 1, and RCF for approximating a given real function on the interval [-1,1]. I will try making tools for transforming these domains later.

## Example
```
>>>CF(np.exp, 1,1,512,30)
(array([0.58955195, 0.99623936]), array([-0.43416584,  1.        ]), 0.08454872594222647)
```

This output corresponds to the rational (0.99623936 + 0.58955195*x)/(1 -0.43416584*x). As an approximation to exp(x) on the unit disc, this gives an estimated maximum error (in the infinity norm sense) of 0.08454872594222647.
In practice, this error estimate is extremely accurate, both compared to the actual error and the error of the true minimax approximation.

```
>>>RCF(np.exp, 5,0,512,30)
(array([1.26606588e+00, 1.13031821e+00, 2.71495340e-01, 4.43368487e-02,
       5.47421205e-03, 5.46136938e-04]), array([1.]), 4.52055119261374e-05)
```

<b> warning: above coefficients are in the Chebyshev basis, I will add an option for the monomial basis later </b>

In this case, the CF polynomial of order 5 is calculated. Using [lolremez](https://github.com/samhocevar/lolremez), the true minimax polynomial can be calculated, which has an error of 4.5205511926115826e-5, matching to around 12 significant figures.

## Known issues

Trying to approximate `np.cos` with nfft being a power of 2 causes errors, the solution to this is to not pick a power of 2 for nfft in these cases.

Currently the calculations are only done in double precision, whereas ideally the resulting coefficients should be correct to double precision at least. I am looking at solutions for this, but numpy rounds the polynomial operations to double precision, and I'm not aware of a library that supports this in higher precision.

## References

The code has been translated from the Matlab implementations described [here](https://people.maths.ox.ac.uk/trefethen/matlabCF.pdf) and [here](https://www.chebfun.org/examples/approx/CF30.html)

Most the 'modern' literature on the topic of CF approximation seems to have been covered by Lloyd N. Trefethen (author of the above Matlab code) in 1984, this presentation gives a good overview of the literature and main results [here](https://people.maths.ox.ac.uk/trefethen//cftalk.pdf).

One tool I would like to highlight is [lolremez](https://github.com/samhocevar/lolremez), which provides a solid implementation of the Remez algorithm for calculating minimax polynomial approximations. There is also a related [blog post](https://lolengine.net/blog/2011/12/21/better-function-approximations) which explains (rather informally) the need for such approximations too.

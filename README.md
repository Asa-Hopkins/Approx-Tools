# ApproxTools

## About The Project

This is a repository for tools for building polynomial and rational approximations to functions on both complex and real domains. The point of this repository is to make such tools more easily accessible, as many such tools are locked behind the paywall of Matlab like Chebfun (which doesn't work in GNU Octave either), or are entirely proprietary (and also locked behind paywalls) like the Remez function in Maple. The few tools that do exist are often difficult to find, difficult to use, have poor performance or are just generally unfinished.

This repository has recently been rewritten to better align with my uses for it, so it does what I need but it's not yet feature complete by any means. Currently implemented here is the Carathéodory-Fejér (CF) method for creating polynomial approximations on the domain [-1,1], and an implementation of Chebyshev expansion / interpolation at the Chebyshev nodes. There is also an implementation of the Remez algorithm, but its performance isn't great so I'd recommend using CF approximation instead where possible. Remez is still useful if you want a weighted error function, for example for minimising relative error. 

For both Chebyshev and CF approximation it is possible to find the polynomial of lowest degree which meets a given error bound, or to give the degree as an input instead. In either case, an estimate of the error will be calculated for the result. 

## Requirements
- [Eigen 3.4+](https://libeigen.gitlab.io/docs/)
- (optional) [OTFFT](http://wwwa.pikara.ne.jp/okojisan/otfft-en/index.html) - I recommend [this fork](https://github.com/joshbarrass/otfftpp) by Josh Barrass which has a fixed and updated build system. 
- (optional) [Spectra](https://spectralib.org/)

## Getting Started
Once Eigen is installed, then simply clone the repo with
`git clone https://github.com/Asa-Hopkins/ApproxTools.git`,
then
`cd ApproxTools`,
and then the benchmark can be built with
`g++ -O3 -march=native benchmark.cpp -o benchmark`.

To use inside your own projects, just place the `Chebyshev.hpp` file in your include path and include it in your file, the same can be done with `Remez.cpp` if you need the Remez algorithm. Note that `Remez.cpp` depends on `Chebyshev.hpp`.

There are also some options at compile time. To use OTFFT instead of KissFFT (the default in Eigen), then add `-Duse_otfft` and link to OTFFT with `-lotfftpp`. When using OTFFT, there is also the option to cache an FFT after the setup is completed, this can be enabled with `-Dfft_cache`. Finally, to enable spectra, add `-Duse_spectra`.

Note that the order of arguments is important, here's the full command I use for compiling.
`g++ -O3 -march=native -Duse_otfft -Dfft_cache benchmark.cpp -lotfftpp -o benchmark`

## Example Usage
For an example on how to use these functions, check benchmark.cpp and the examples folder. I also use this in my continuous time quantum walk simulator, [MultiStageQW](https://github.com/Asa-Hopkins/MultiStageQW), so you could consider that an example. 
I will be adding better documentation later but I've tried to keep the code readable.

## Known Issues and To-Do

Main items on the to-do list are to add support for higher precision (long doubles or __float128 ideally), add Python bindings and also add rational CF approximation and complex CF approximation.

Since it's difficult to bound L_inf error of the FFT, the current method cannot tell if a value below around 10*eps*n is truly 0 or not, so this is the criteria for convergence. For double precision needing 1024 terms, this gives a limit of around 1e-11, which is much worse than the actual accuracy that should be possible. When nearing this limit, RCF fares better but at the cost of higher runtime. Enabling Spectra mitigates this but it still isn't ideal.

The current method for error estimation for Chebyshev is to expand to more terms than necessary and then add up the magnitudes of the extra terms to get an upper bound. For smooth functions this works well, but for non-smooth functions this can miss the effects of a slowly decaying tail, so this needs some improvement. When trying to expand `abs(x)`, the error estimate is around half the real error value. For `sin(abs(x))^3` this works fine, so the function doesn't need to be that smooth.

It is possible to create pathological functions that will break parts of the algorithm, e.g trying to expand `sin(T_6(x))` will fail, as the way that convergence is checked will be tricked by a series with more than 5 zeros in a row.

I believe it's possible to make the Remez algorithm run in `O(N log^2(N))` time, but this would require writing a fast multi-point evaluation algorithm to make root finding faster, and also a fast interpolation method to speed up solving the Vandermonde-like matrix that appears. I think both can be done using a non-uniform FFT and its inverse, but I haven't tried yet. Failing this, there are other known methods that achieve this time bound (e.g [here](https://arxiv.org/pdf/1304.8069)).

## Contributing
I am open to contributions, discussions, criticism and feature requests. I am genuinely interested in knowing any use cases for this work and extending it to fit them where it's within my ability.

## References

The old python version of the code was translated from the Matlab implementations described [here](https://people.maths.ox.ac.uk/trefethen/matlabCF.pdf) and [here](https://www.chebfun.org/examples/approx/CF30.html), however the new C++ version has been written from scratch and has performance improvements over a direct translation.

Most the modern literature on the topic of CF approximation seems to have been covered by Lloyd Trefethen (author of the above Matlab code) and Martin Gutknecht in the early 1980s, this presentation gives a good overview of the literature and main results [here](https://people.maths.ox.ac.uk/trefethen//cftalk.pdf). The CF function in this repository is based on [this paper](https://epubs.siam.org/doi/abs/10.1137/0719022), and the rational extension will be based on [this paper](https://www.jstor.org/stable/2157229). I also recommend Trefethen's book "Approximation theory and approximation practice" which covers these topics in great detail.

An FFT based method for Chebyshev interpolation in the Chebyshev basis was first described by [Ahmed and Fisher](http://dx.doi.org/10.1080/00207167008803043), but I now use a more efficient method described by [Makhoul](https://ieeexplore.ieee.org/document/1163351).

I use one of the methods described in "[Bit Twiddling Hacks](https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2)" by Sean Eron Anderson to get the smallest power of two larger than a given input. In my tests this is faster than a naive approach even on modern hardware, not that it's actually a bottleneck.

For multiplication of Chebyshev polynomials I use a method described by Pascal Giorgi in "On Polynomial Multiplication in Chebyshev Basis". He provides C++ code for it but there is a mistake in the 2010 version, I recommend reading the [2013 version on arXiv](https://arxiv.org/abs/1009.4597). I have modified it to work with inputs of differing degrees and also using T_0 = 1/2 as opposed to T_0 = 1.

For error analysis of the FFT I'm using results from "Error Analysis of Some Operations Involved in the Cooley-Tukey Fast Fourier Transform" by Brisebarre et al. The L_inf bounds are awful so I'm hoping it's possible to tighten it with assumptions of smoothness or something.

For finding roots of a Chebyshev polynomial I set up a colleague matrix and use a fast QR iteration described by Serkh and Rokhlin in "A Provably Componentwise Backward Stable O(n2) QR Algorithm for the Diagonalization of Colleague Matrices".

I also do a step of Rayleigh-Ritz iteration and use a result from Higham's book "Accuracy and Stability of Numerical Algorithms" in order to solve a quadratic equation in a numerically stable way.

## Acknowledgments

One tool I would like to highlight is [lolremez](https://github.com/samhocevar/lolremez), which provides a reasonable implementation of the Remez algorithm for calculating minimax polynomial approximations, and was my first introduction to approximation theory. I recommend the related [blog posts](https://lolengine.net/blog/2011/12/21/better-function-approximations) which explains the need for such approximations, and also the git wiki which explains some tricks for creating approximations. The blog is dead now but it's still readable on archive.org.

Thanks to Josh for helping me fix OTFFT too, it has some advantages over other options that make it a good fit for this project.

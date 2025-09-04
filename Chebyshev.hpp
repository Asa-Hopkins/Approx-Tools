#pragma once
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/FFT>
#include <vector>
#include <complex>
#include <cmath>
#include <iostream>

template<typename Scalar = double>
class Chebyshev {
public:
  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using Complex = std::complex<Scalar>;
  using VectorC = Eigen::Matrix<Complex, Eigen::Dynamic, 1>;
  using Matrix2C = Eigen::Matrix<Complex, 2, 2>;

  Vector coeffs;
  int degree;

  // Constructor
  Chebyshev() : coeffs(Vector::Zero(1)), degree(0) {}
  Chebyshev(const Vector &coefficients)
    : coeffs(coefficients), degree(static_cast<int>(coefficients.size()) - 1) {}

  // Print coefficients
  friend std::ostream &operator<<(std::ostream &os, const Chebyshev &c) {
    os << c.coeffs.transpose();
    return os;
  }

  // Addition
  Chebyshev operator+(const Chebyshev &q) const {
    if (degree >= q.degree) {
      Vector c = coeffs;
      c.head(q.degree + 1) += q.coeffs;
      return Chebyshev(c);
    } else {
      Vector c = q.coeffs;
      c.head(degree + 1) += coeffs;
      return Chebyshev(c);
    }
  }

  // Negation
  Chebyshev operator-() const {
    return Chebyshev(-coeffs);
  }

  // Subtraction
  Chebyshev operator-(const Chebyshev &q) const {
    return *this + (-q);
  }  

  // Evaluate at a single point using Clenshaw algorithm
  Scalar eval(Scalar x) const {
    int n = degree + 1;
    if (n <= 0) return Scalar(0);
    if (n == 1) return coeffs[0];

    Scalar b2 = Scalar(0);
    Scalar b1 = coeffs[n - 1];

    for (int r = n - 2; r > 0; --r) {
      Scalar tmp = b1;
      b1 = Scalar(2) * x * b1 - b2 + coeffs[r];
      b2 = tmp;
    }
    return x * b1 - b2 + coeffs[0];
  }

  // Allow function-call syntax for evaluation too
  Scalar operator()(Scalar x) const {
    return eval(x);
  }

  // Evaluate on a vector of points
  // Naive way for now
  Vector eval(const Vector &xs) const {
    Vector ys(xs.size());
    for (int i = 0; i < xs.size(); ++i) {
      ys[i] = eval(xs[i]);
    }
    return ys;
  }

  // returns Chebyshev object for derivative
  // Uses dT_n/dx = n*U_{n-1}, and U_n = 2*T_n + U_{n - 2}
  // So, essentially replace U_n with 2*T_n, and add to U_{n - 2} term
  Chebyshev deriv() const {
    if (degree <= 0) {
      Vector zero = Vector::Zero(1);
      return Chebyshev(zero);
    }

    Vector new_coeffs = Vector::Zero(degree);
    int n = degree;

    for (int i = n - 1; i >= 0; --i) {
      new_coeffs[i] = 2 * (i + 1) * coeffs[i + 1];
      if (i + 2 < n) {
        new_coeffs[i] += new_coeffs[i + 2];
      }
    }
    //T_1 is just x, so its derivative is U_0 and not 2*U_0
    new_coeffs[0] /= 2;
    return Chebyshev(new_coeffs);
  }

  // FFT-based fit specialized for f: [-1,1] -> R functions
  // See "Study of algorithmic properties of chebyshev coefficients" by Ahmed and Fisher, equation 6
  // Computes Chebyshev expansion (by default, option for interpolant) of f(x) on [-1,1] to order n

  // It's actually better (more accurate and more consistent timings) to calculate to a higher degree than requested and truncate
  // See chapter 4 of Approximation Theory and Approximation Practice (By Trefethen) for intro on aliasing issues and truncation
  // Theorem 16.1 says that going to higher n and truncating has a Lebesque constant that's reduced by factor pi/2

  template <typename Func>
  static Chebyshev fit(Func f, unsigned int n, bool truncate = true) {
    if (n == 0) {
      Vector zero(1);
      zero << Scalar(0);
      return Chebyshev(zero);
    }

    int m = n;

    //Taken from "Bit Twiddling Hacks" by Sean Eron Anderson
    //rounds m to the next power of 2 greater or equal to it
    if (truncate){
      m--;
      m |= m >> 1;
      m |= m >> 2;
      m |= m >> 4;
      m |= m >> 8;
      m |= m >> 16;
      m++;
    }

    const Scalar pi = std::acos(Scalar(-1));
    const Complex I(Scalar(0), Scalar(1));
    Eigen::FFT<Scalar> fft;

    // Buffers for FFT input and output
    std::vector<Complex> vals(2*m, Complex(0,0));
    std::vector<Complex> fft_out(2*m);

    // Build array of f(cos(theta_k)) values
    // We use 0-based indexing compared to reference
    for (unsigned int i = 0; i < m; ++i) {
      Scalar theta = (Scalar(2*i + 1) * pi) / (Scalar(2)*m);
      Scalar x = std::cos(theta);
      vals[i] = Complex(f(x), Scalar(0));
    }

    // Inverse FFT
    fft.inv(fft_out, vals);

    // Scale by 4*exp(i*pi*k/(2n)) and truncate to first n
    // iFFT divides by the length N = 2m, so the 2/m factor
    // in the reference becomes 4 for us
    Vector coeffs(n);
    for (unsigned int k = 0; k < n; ++k) {
      Scalar angle = (pi * Scalar(k)) / (Scalar(2) * m);
      Complex mult = std::exp(I * angle) * Scalar(4);
      Complex val = fft_out[k] * mult;
      coeffs[k] = val.real();
    }

    coeffs[0] /= Scalar(2);
    return Chebyshev(coeffs);
  }
  
  //Fit polynomial at arbitrary nodes
  //O(n^2) method - set up Lagrange interpolant, evaluate at Chebyshev nodes
  //then call the FFT fit above. 


  //Finds roots of Chebyshev series polynomial
  //Constructs the "colleague matrix" which has our polynomial as its characteristic polynomial
  //Then exploits its nearly tridiagonal form to find eigenvalues with fast QR iterations
  //I implement the stable QR algorithm from Serkh and Rokhlin (2021)
  //Good convergence - around 2n iterations needed consistently

  // ================= Algorithm 2: Rotate =================
  static void
  rotate(Eigen::Ref<VectorC> d, Eigen::Ref<VectorC> b, Eigen::Ref<VectorC> p, Eigen::Ref<VectorC> q, Eigen::Ref<VectorC> g,
    const std::vector<Matrix2C> &rotations) 
    {
      int n = p.size();

      for (int k = n - 1; k > 0; --k) {
        Matrix2C Qk = rotations[k - 1].conjugate();

        auto temp = Qk(0,0)*d[k-1] - Qk(0,1)*p[k-1] * std::conj(q[k]);
        b[k-1] = Qk(1,0)*d[k-1] - Qk(1,1)*p[k-1] * std::conj(q[k]);
        d[k-1] = temp;

        d[k] = Qk(1,0)*g[k-1] + Qk(1,1)*d[k];

	Qk = Qk.conjugate();

        temp = Qk(0,0)*q[k-1] + Qk(0,1)*q[k];
        q[k] = Qk(1,0)*q[k-1] + Qk(1,1)*q[k];
        q[k-1] = temp;
      }

      return;
  }


  // ================= Algorithm 1: Elimination =================
  static void
  elimination(Eigen::Ref<VectorC> d, Eigen::Ref<VectorC> b, Eigen::Ref<VectorC> p, Eigen::Ref<VectorC> q, Eigen::Ref<VectorC> g, std::vector<Matrix2C> &rotations)
  {
    int n = p.size();
    g = b.conjugate();
    static thread_local VectorC q2 = q;

    for (int k = n - 1; k > 0; --k) {
      Complex s = b[k - 1] + p[k - 1]*std::conj(q[k]);
      Complex c = d[k]     + p[k]*std::conj(q[k]);

      Scalar h = std::sqrt((c*std::conj(c) + s*std::conj(s)).real());

      Matrix2C Qk;
      if (h == 0) {
        Qk.setIdentity();
      } else {
        Qk << c/h, -s/h,
              std::conj(s)/h, std::conj(c)/h;
      }
      rotations[k-1] = Qk;

      if (k != 1) {
        g[k-2] = Qk(0,0)*g[k-2] - Qk(0,1)*q2[k]*std::conj(p[k-2]);
      }

      auto temp = Qk(0,0)*d[k-1] + Qk(0,1)*g[k-1];
      g[k-1] = Qk(1,0)*d[k-1] + Qk(1,1)*g[k-1];
      d[k-1] = temp;

      temp = Qk(0,0)*b[k-1] + Qk(0,1)*d[k];
      d[k] = Qk(1,0)*b[k-1] + Qk(1,1)*d[k];
      b[k-1] = temp;
 
      temp = Qk(0,0)*p[k-1] + Qk(0,1)*p[k];
      p[k] = Qk(1,0)*p[k-1] + Qk(1,1)*p[k];
      p[k-1] = temp;
      if (std::norm(p[k-1]*std::conj(q[k])) +
      std::norm(p[k]*std::conj(q[k])) >
      std::norm(b[k-1]) + std::norm(d[k])) {
        p[k-1] = -b[k-1]/std::conj(q[k]);
      }

      temp = Qk(0,0)*q2[k-1] + Qk(0,1)*q2[k];
      q2[k] = Qk(1,0)*q2[k-1] + Qk(1,1)*q2[k];
      q2[k-1] = temp;
    } 

    return;
  }

  // ================= Algorithm 4: Shifted QR iteration =================
  static VectorC QR(VectorC d, VectorC b, VectorC p, VectorC q, Scalar eps = 1e-10) {
    //pre-allocate as much as possible
    Eigen::Matrix<Complex,2,2> A;
    int n = p.size();
    std::vector<Matrix2C> rotations;
    rotations.resize(n - 1);
    VectorC g = b.conjugate();

    for (int i = 0; i < n - 1; ++i) {
      Complex mu_sum(0,0);

      while (std::abs(b[i] + p[i]*std::conj(q[i+1])) > eps) {
        //Take a 2x2 section of our matrix and use its eigenvalues as our shift amount
        A << d[i] + p[i]*std::conj(q[i]),      b[i] + p[i]*std::conj(q[i+1]),
             std::conj(b[i]) + p[i+1]*std::conj(q[i]), d[i+1] + p[i+1]*std::conj(q[i+1]);

        Complex tr = A(0,0) + A(1,1);
        Complex det = A(0,0)*A(1,1) - A(1,0)*A(0,1);

        Complex sq = std::sqrt(tr*tr - Scalar(4)*det);
        Complex mu1 = Scalar(0.5)*(tr + sq);
        Complex mu2 = Scalar(0.5)*(tr - sq);

	//pick the one that A(0,0) is closest to
        Complex mu = (std::abs(A(0,0) - mu1) < std::abs(A(0,0) - mu2)) ? mu1 : mu2;

        mu_sum += mu;
        d.tail(n - i).array() -= mu;

        //Perform a QR step
        elimination(d.tail(n-i), b.tail(n-i-1), p.tail(n-i), q.tail(n-i), g.tail(n-i-1), rotations);
        rotate(d.tail(n-i), b.tail(n-i-1), p.tail(n-i), q.tail(n-i), g.tail(n-i-1), rotations);
      }

      d.tail(n - i).array() += mu_sum;
    }
    return d + p.cwiseProduct(q.conjugate());
  }

  // ================= Root finder =================
  VectorC roots() {

    int d = degree;
    if (d <= 0) return VectorC();

    // nth unit vector
    Vector e_n = Vector::Zero(d);
    e_n[d-1] = 1;

    // coeffs (normalised to monic)
    Vector c = coeffs / coeffs[degree];
    c = -c / Scalar(2);
    c[0] *= std::sqrt(Scalar(2));

    // superdiagonal
    Vector a = Vector::Constant(d-1, Scalar(0.5));
    a[0] *= std::sqrt(Scalar(2));

    // diagonal
    Vector diag = Vector::Zero(d);

    return QR(
      diag.template cast<Complex>(),
      a.template cast<Complex>(),
      e_n.template cast<Complex>(),
      c.head(d).template cast<Complex>()
    );

  }

  //Uses the method from "Real Polynomial Chebyshev Approximation by the Carathéodory-Fejér Method"
  //By Gutknecht and Trefethen (1982)

  //The algorithm in the above paper is simpler than their 1983 one which covers rational approximation
  template <typename Func>
  static Chebyshev RCF(Func f, unsigned int n, unsigned int N) {
  //inputs: f is a function to be approximated on [-1,1]
  //n is the returned polynomial degree
  //N is the polynomial degree for calculations, the error of fit(f,N) should be negligible compared to fit(f,n)

  //Basic idea is to form a Hankel matrix from an extended Chebyshev series of the function
  //Then the largest eigenvalue estimates the approximation error and the eigenvector describes
  //a rational function for the error, called b(x)
  //We use the knowledge of that error to correct the Chebyshev series of order n

  //Some notes on optimisation
  //The 1983 paper suggests multiple places where FFTs could be applied but I haven't implemented it yet
  //They refer to "FAST FOURIER METHODS IN COMPUTATIONAL COMPLEX ANALYSIS" by Peter Henrici (1979)

  //The laurent series for the Blaskche product b(x) can be calculated via FFT, since it's the ratio
  //of two polynomials in the monomial basis, and multiplication/division is pointwise in fourier space.

  //Forming the polynomial q(x) which is p(x) but with the zeros inside the unit disc factored out
  //can be done with an FFT method too, see section 3.2 of Peter's book
  //The algorithm there seems to give p(x)/q(x) and I haven't figured out a way to get q(x) directly

  //Something they don't mention is that mutliplication by a Hankel matrix can be done with an FFT too
  //This would make the fastest way to get the largest eigenvalue/vector a Lanczos iteration using these FFTs
  //I've implemented this in Python but not C++ yet, it is really fast even for small n

  //I don't bother with any of that yet. For smooth cases, N - n will be small so just using direct methods is fine.
  
    using Eigen::seq;
    using Eigen::last;

    if (n == 0) {
      Vector zero(1);
      zero << Scalar(0);
      return Chebyshev(zero);
    }

    auto c = Chebyshev::fit(f, N+1);

    // construct Hankel matrix of size (N-n) x (N-n)
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> H = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(N - n, N - n);

    for (int i = 0; i < N-n; i++){
      H(i,Eigen::seq(0,last - i)) = c.coeffs(seq(i+n+1,last));
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> es(H);
    auto evals = es.eigenvalues();            // Should be ascending order but we do a full search anyway
    auto evecs = es.eigenvectors();           // columns are eigenvectors

    //We want argmax of abs(evals)
    Eigen::Index idx;
    (evals.array().abs()).maxCoeff(&idx);

    Scalar val = evals(idx);
    Eigen::VectorX<Scalar> u = evecs.col(idx).transpose();

    //Let v(z) be the polynomial with u[1:]/u[0] as its coefficients
    //We want the Laurent series (outside unit disc) of val * z^N * v(z)/v(z*)
    //Only need coefficients b_k for orders -n to n

    //We essentially use the long division algorithm to get this iterative formula
    //b_k given by c_k for n+1 <= k <= N
    //Then the others are given by b_k = -1/u_1 ( b_{k+1} * u_2 + b_{k+1} * u_3 + ... + b_{k+M-m-1} * u_{M-m))
    Vector b = Vector::Zero(n + N + 1);
    
    b(seq(2*n + 1, last)) = c.coeffs(seq(n + 1, last));
    
    for (int k = n; k + n + 1 > 0; k--){
      int i = k + n;
      b[i] = -1/u[0] * b(seq(i+1,i+N-n-1)).dot(u(seq(1,last)));
      c.coeffs[abs(k)] -= b[i];
    }
    
    c.coeffs.conservativeResize(n+1);
    c.degree = n;

    return c;
  }

};
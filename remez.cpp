#include "Chebyshev.hpp"
#include <chrono>
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>

template<typename Scalar>
struct Remez{
  using Vector  = Eigen::Array<Scalar, Eigen::Dynamic, 1>;
  using Complex = std::complex<Scalar>;
  using VectorC = Eigen::Array<Complex, Eigen::Dynamic, 1>;
  using Poly    = Chebyshev<Scalar>;

  // -------------------------
  // Barycentric interpolation
  // -------------------------
  static std::function<Scalar(Scalar)> barycentric_interpolator(const Vector& xi, const Vector& yi){
    const int n = xi.size();
    
    // Compute barycentric weights
    Vector w(n);
    for (int j = 0; j < n; ++j) {
        //1/prod blows up so we double every value as a preconditioner.
        //I'm unsure why 2 works, the geometric average spacing of Chebyshev nodes is pi/n
        Vector diff = 2*(xi[j] - xi);
        diff[j] = 1.0;  // avoid zero for self-subtraction
        w[j] = 1.0 / diff.prod();
    }

    // Return interpolator as scalar function
    return [xi, yi, w, n](Scalar z) -> Scalar {
        // Check if z equals one of the xi (within tolerance)
        for (int j = 0; j < n; ++j) {
            if (std::abs(z - xi[j]) < 1e-14) {
                return yi[j];
            }
        }

        Vector diff = z - xi;
        Vector numer = (w * yi) / diff;
        Vector denom = w / diff;

        return numer.sum() / denom.sum();
    };
  }

  static Poly SimpleRemez(std::function<Scalar(Scalar)> f, int n, int N, std::function<Scalar(Scalar)> w){

    //If w has roots on [-1,1], then the error there must be 0 else the relative error is infinite
    //This means those roots also exist in the minimax polynomial, which we can enforce exactly
    Poly weight = Poly::fit(w, N+1);
    weight.trunc();

    VectorC roots = weight.roots();

    //We need to separate the roots inside and outside of [-1,1]
    //Check Chebyshev.hpp for notes on a faster algorithm for this
    VectorC sorted_roots = roots*0;
    int roots_in = 0;
    for (int i = 0; i < weight.degree; i++){
      Complex a = roots[i];
      if ((abs(a) < 1) and (abs(imag(a)) < n*1e-15)){
        sorted_roots[roots_in] = a;
        roots_in += 1;
      } else {
      sorted_roots[weight.degree - 1 + roots_in - i] = a;
      }
    }

    if (roots_in > 0){
      //These roots must exist in the output
      Poly forced_roots = Poly::from_roots(sorted_roots.head(roots_in));

      Poly new_weight = Poly::from_roots(sorted_roots.tail(weight.degree - roots_in));
      
      Poly new_remez = Remez::SimpleRemez([&](Scalar a) -> Scalar {return f(a)/forced_roots(a); }, n - roots_in, N, new_weight);
      return new_remez*forced_roots;
    }

    //Polynomial approximations of f and 1/w, N should be large enough that these are good approximations  
    Poly w_i = Poly::fit([&](Scalar a) -> Scalar {return 1/w(a); }, N+1);
    w_i.trunc();

    Poly f_c = Poly::fit(f, N+1);

    //This is going to become the minimax polynomial of order n
    Poly p2 = Poly(f_c.coeffs.head(n+1));

    //There are n+2 extrema but the last one is always x = 1 so we don't keep it in the array
    Vector extreme_x = Vector::Zero(n+1);
    Vector extreme_y = Vector::Zero(n+1);
    extreme_x[0] = -1;
    Scalar last_error = 0;

    for (int i = 0; i < 20; i++){
      Poly error = (f_c - p2)*w_i;
      //The roots function divides by error.coeffs[-1] so we make sure there's no trailing zeros
      error.trunc();
      VectorC extrema = error.deriv().roots();


      int j = 1;
      for (const auto& a: extrema){
        //If this root is on [-1,1] then it's one of the oscillation positions
        if ((abs(a) < 1) and (abs(imag(a)) < n*1e-15)){
          extreme_x[j] = real(a);
          j+=1;
        }
      }

      //std::cout << extreme_x;

      //Sort the extreme values since they must oscillate
      std::sort(extreme_x.data(), extreme_x.data() + extreme_x.size());

      Scalar temp = 1;
      for (j = 0; j < n + 1; j++){
        extreme_y[j] = temp*w(extreme_x[j]);
        temp *= -1;
      }

      auto p1 = barycentric_interpolator(extreme_x, extreme_y);
      p2 = Poly::fit(barycentric_interpolator(extreme_x, extreme_y), n+1);

      Scalar L22 = temp*w(1) - p2.coeffs.sum();

      Vector f_i = Vector::Zero(n+1);
      for (j = 0; j < n + 1; j++){
        f_i[j] = f(extreme_x[j]);
      }

      Poly p3 = Poly::fit(barycentric_interpolator(extreme_x, f_i), n+1);

      //This is the estimated minimax error 
      Scalar y2 = (f(1) - p3.coeffs.sum())/L22;

      p2 = Poly(p3.coeffs - y2*p2.coeffs);

      if (abs((last_error - y2)/y2) < 1e-8){break;}

      last_error = y2;
    }
    return p2; 
  }
};
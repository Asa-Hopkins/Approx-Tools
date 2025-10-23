#include "../Chebyshev.hpp"
#include <iostream>
#include <iomanip>

using Scalar = double;

Scalar example1(Scalar x){
  return exp(-5*x*x);
}

Scalar example2(Scalar x){
  //An example with two continuous derivatives
  Scalar y = sin(abs(x));
  return y*y*y;
}

int main() {
  using Poly = Chebyshev<Scalar>;
  
  double tolerance = 1e-8;
  Poly p0 = Poly::fit_bounded(example1, tolerance);
  Poly r0 = Poly::RCF_bounded(example1, tolerance);

  std::cout << "Example 1: exp(-5*x*x) \n";
  std::cout << std::setprecision (10) << "\n";

  std::cout << "Estimated error of bounded Chebyshev:" << (double)p0.error << " with degree " << p0.degree << "\n";
  std::cout << "Estimated error of bounded RCF: " << (double)r0.error << " with degree " << r0.degree << "\n";
  std::cout << "Measured error of bounded RCF is " << (double)example1(-1) - (double)r0(-1) << " " << (double)example1(1) - (double)r0(1) << "\n\n\n";

  p0 = Poly::fit_bounded(example2, tolerance);
  r0 = Poly::RCF_bounded(example2, tolerance);

  std::cout << "Example 2: sin(abs(x))^3 \n\n";

  std::cout << "Estimated error of bounded Chebyshev:" << (double)p0.error << " with degree " << p0.degree << "\n";
  std::cout << "Estimated error of bounded RCF: " << (double)r0.error << " with degree " << r0.degree << "\n";
  std::cout << "Measured error of bounded RCF is " << (double)example2(-1) - (double)r0(-1) << " " << (double)example2(1) - (double)r0(1) << "\n\n\n";

  return 0;
}

#include "Chebyshev.hpp"
#include "remez.cpp"
#include <iostream>
#include <chrono>
#include <iomanip>

using Scalar = double;

int main() {
    using Poly = Chebyshev<Scalar>;
    using clock = std::chrono::high_resolution_clock;

    int n = 10;
    int N = 20;
    double scale = 1.0;
    for (int j = 0; j < 11; j++){
    n = (scale*12)/10 + 20;

    auto f = [scale](Scalar x) {
        return sin(scale*x) + cos(scale*x);
    };

    //This weight function gives us the infinity norm
    auto w = [](Scalar x){
      return 1.0;
    };

    Poly p3 = Poly::fit(f, n+1);

    int max_n = 0;
    double err_est = 0;
    for (int i = 0; i < n; i++){
      err_est += abs(p3.coeffs[i]);
      if (abs(p3.coeffs[i]) > 1e-6){
        max_n = i;
        err_est = 0;
      }
    }

    auto benchmark_fit = [&](int i) {
        //no optimising away
        volatile double guard = 0;
        auto t1 = clock::now();
        Poly p = Poly::fit(f, max_n+1);
        auto t2 = clock::now();
        guard += p(i);
        return std::chrono::duration<double>(t2 - t1).count();
    };

    auto benchmark_RCF = [&](int i) {
        volatile double guard = 0;
        auto t1 = clock::now();
        Poly p = Poly::RCF(f, max_n, N+max_n);
        auto t2 = clock::now();
        guard += p(i);
        return std::chrono::duration<double>(t2 - t1).count();
    };

    auto benchmark_Remez = [&](int i) {
        volatile double guard = 0;
        auto t1 = clock::now();
        Poly p = Remez<Scalar>::SimpleRemez(f, max_n, N+max_n,w);
        auto t2 = clock::now();
        guard += p(i);
        return std::chrono::duration<double>(t2 - t1).count();
    };

    // warm-up
    volatile double sink = 0;
    sink += Poly::fit(f, 10)(0.1);
    sink += Poly::RCF(f, 10, 100)(0.1);


    double total = 0;
    int runs = 10;
    for (int i = 0; i < runs; ++i) {
        total += benchmark_fit(i);
    }

    std::cout << "avg fit() time: " << total / runs << " seconds\n";

    total = 0;
    for (int i = 0; i < runs; ++i) {
        total += benchmark_RCF(i);
    }

    std::cout << "avg RCF() time: " << total / runs << " seconds\n";

    total = 0;
    for (int i = 0; i < runs; ++i) {
        total += benchmark_Remez(i);
    }

    std::cout << "avg Remez() time: " << total / runs << " seconds\n";

    Poly p = Poly::fit(f, max_n+1);
    Poly p2 = Poly::RCF(f, max_n, max_n+N);
    Poly p4 = Remez<Scalar>::SimpleRemez(f, max_n, max_n + N, w);
    //std::cout << std::setprecision (10) << 5.0 << "\n\n";
    //Poly p4 = Remez<Scalar>::SimpleRemez(f, max_n, max_n + N, f);
    
    // estimate max error
    double max1 = 0;
    double max2 = 0;

    max2 = abs(p2(-1) - f(-1));
    max1 = abs(p4(-1) - f(-1));

    std::cout << "x-axis scaled by:" << scale << ", Number of terms:" << max_n << "\n";
//    std::cout << "Max error of Chebyshev: " << max1 << "\n";
    std::cout << "estimated error of Chebyshev: " << err_est << "\n";
    std::cout << "Max error of RCF: " << max2 << "\n";
    std::cout << "Max error of Remez: " << max1 << "\n";
    std::cout << "RCF is " << err_est/max2 << "x more accurate\n";

    int extra = 0;
    while (err_est > max2){
      err_est -= abs(p3.coeffs[max_n + 1 + extra]);
      extra += 1;
    }

    std::cout << "estimated extra terms needed for Chebyshev: " << extra << " " << "\n";

    std::cout << "\n\n";

    scale *= 2;

    }

    return 0;
}

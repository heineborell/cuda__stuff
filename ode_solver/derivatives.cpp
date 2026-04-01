#include "helper.h"
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <vector>

template <typename T> void printArray(std::vector<T> const &arr) {
  for (T const &item : arr) {
    std::cout << item << '\n';
  }
}

std::vector<double> vectorScale(std::vector<double> &arr, double scalar) {
  std::vector<double> scaled;
  scaled.reserve(size(arr));
  for (double &item : arr)
    scaled.push_back(scalar * item);
  return scaled;
}

// matrix multiply AB=C
std::vector<double> mmCpu(std::vector<double> const &A,
                          std::vector<double> const &B, int Nay, int Nbx,
                          int K) {
  // C dimensions: rows of A (Nay) x cols of B (Nbx) , K is common dim
  std::vector<double> C(static_cast<std::size_t>(Nay * Nbx), 0.0);

  for (int y = 0; y < Nay; ++y) {
    for (int x = 0; x < Nbx; ++x) {
      double sum = 0.0;
      for (int i = 0; i < K; ++i) {
        sum += A.data()[y * K + i] * B.data()[i * Nbx + x];
      }
      C.data()[y * Nbx + x] = sum;
    }
  }
  return C;
}

// vector addition
template <typename T>
std::vector<T> vecAddCpu(std::vector<T> const &x, std::vector<T> const &y) {

  std::size_t N{x.size()};
  std::vector<T> z(N, 0.0);
  for (std::size_t i{0}; i < N; ++i)
    z[i] = x[i] + y[i];
  return z;
}

// vector subtract
template <typename T>
std::vector<T> vecSubtractCpu(std::vector<T> const &x,
                              std::vector<T> const &y) {

  std::size_t N{x.size()};
  std::vector<T> z(N, 0.0);
  for (std::size_t i{0}; i < N; ++i)
    z[i] = x[i] - y[i];
  return z;
}

template <typename T> void timeVec(std::vector<T> &vec, T dt) {
  int N{static_cast<int>(vec.size())};
  for (std::size_t i{0}; i < N; ++i)
    vec[i + 1] = vec[i] + dt;
}

template <typename T>
void evaluate(std::vector<T> &fArray, std::function<T(T)> &f,
              std::vector<T> &timeArray) {
  std::size_t N{fArray.size()};
  for (std::size_t i{0}; i < N; ++i) {
    fArray[i] = f(timeArray[i]);
  }
}

template <typename T>
void derivativeForward(std::vector<T> &df, std::function<T(T)> &f,
                       std::vector<T> &timeArray, T dt) {
  std::size_t N{df.size()};
  for (std::size_t i{0}; i < N; ++i) {
    df[i] = (f(timeArray[i + 1]) - f(timeArray[i])) / dt;
  }
}

template <typename T>
void derivativeCenter(std::vector<T> &df, std::function<T(T)> &f,
                      std::vector<T> &timeArray, T dt) {
  std::size_t N{df.size()};
  for (std::size_t i{0}; i < N; ++i) {
    df[i] = (f(timeArray[i + 1]) - f(timeArray[i - 1])) / (2 * dt);
  }
}
int main() {
  Timer t;

  using Real = double;
  Real dt{1E-3}; // time step
  Real T{1};     // total time
  std::function<Real(Real)> fn{[](Real t) { return t * t * t; }};
  std::function<Real(Real)> dfn{[](Real t) { return 3 * t * t; }};
  int N{static_cast<int>(T / dt)};
  std::vector<Real> time(N, 0.0);
  std::vector<Real> dfForward(N, 0.0);
  std::vector<Real> dfCenter(N, 0.0);
  std::vector<Real> dfExact(N, 0.0);
  timeVec(time, dt);
  derivativeForward(dfForward, fn, time, dt);
  derivativeCenter(dfCenter, fn, time, dt);
  evaluate(dfExact, dfn, time);
  // printArray(df);
  std::vector<Real> resultForward{vecSubtractCpu(dfExact, dfForward)};
  std::vector<Real> resultCenter{vecSubtractCpu(dfExact, dfCenter)};
  std::cout << std::abs(resultForward.back()) << " Forward Derivative error "
            << '\n';
  std::cout << std::abs(resultCenter.back()) << " Forward Center error "
            << '\n';
  // printArray(time);

  std::cout << "total time is " << t.elapsed() << '\n';
  return 0;
}

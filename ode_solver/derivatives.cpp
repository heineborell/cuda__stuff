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
              std::vector<T> &timeArray, T dt) {
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

int main() {
  Timer t;

  float dt{0.01f}; // time step
  float T{10.0f};  // total time
  std::function<float(float)> fn{[](float t) { return t * t * t; }};
  std::function<float(float)> dfn{[](float t) { return 3 * t * t; }};
  int N{static_cast<int>(T / dt)};
  std::vector<float> time(N, 0.0f);
  std::vector<float> df(N, 0.0f);
  std::vector<float> dfExact(N, 0.0f);
  timeVec(time, dt);
  derivativeForward(df, fn, time, dt);
  evaluate(dfExact, dfn, time, dt);
  printArray(vecSubtractCpu(dfExact, df));
  // printArray(dfExact);

  std::cout << "total time is " << t.elapsed() << '\n';
  return 0;
}

#include "Random.h"
#include "helper.h"
#include "math.h"
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

void printArray(std::vector<double> const &arr) {
  for (double const &item : arr) {
    std::cout << item << '\n';
  }
}

std::vector<float> castArray(std::vector<double> &arr) {
  std::size_t N{arr.size()};
  std::vector<float> A(N, 0.0f);
  for (std::size_t i{0}; i < N; ++i) {
    A[i] = static_cast<float>(arr[i]);
  }
  return A;
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

// vectorAdd addition
std::vector<double> vecAddCpu(std::vector<double> const &x,
                              std::vector<double> const &y) {

  std::size_t N{x.size()};
  std::vector<double> z(N, 0.0);
  for (std::size_t i{0}; i < N; ++i)
    z[i] = x[i] + y[i];
  return z;
}

void timeVec(std::vector<double> &vec, int gridDimX, int gridDimY, double dt) {
  int N{static_cast<int>(vec.size())};
  for (int i{gridDimX * (gridDimY - 1)}; i < N; ++i)
    vec.data()[i + 1] = vec.data()[i] + dt;
}

void showMatrix(std::vector<double> &vec, int gridDimX, int gridDimY) {
  int N{static_cast<int>(vec.size())};
  std::cout << std::setprecision(3);
  for (int i{0}; i < N; ++i) {
    if (i % gridDimX == 0)
      std::cout << '\n';
    std::cout << vec[i] << " ";
  }
}

void forwardEuler(
    std::vector<double> &X,
    std::vector<std::function<double(double, double, double)>> &rhs, double dt,
    std::size_t dimX, std::size_t dimY) {

  for (std::size_t x{0}; x < dimX - 1; ++x) {
    for (std::size_t y{0}; y < dimY - 1; ++y) {
      X[y * dimX + (x + 1)] =
          X[y * dimX + x] +
          dt * rhs[y](X[0 * dimX + x], X[dimX + x], X[2 * dimX + x]);
    }
  }
}

void rungeKutta2Order(
    std::vector<double> &X,
    std::vector<std::function<double(double, double, double)>> &rhs, double dt,
    std::size_t dimX, std::size_t dimY) {

  std::vector<double> X1(dimX * dimY, 0.0);
  for (std::size_t x{0}; x < dimX - 1; ++x) {
    for (std::size_t y{0}; y < dimY - 1; ++y) {
      X1[y * dimX + (x + 1)] =
          X[y * dimX + x] + dt / 2 *
                                rhs[y](X[0 * dimX + x], X[dimX + x],
                                       X[2 * dimX + x]); // half-step
      X[y * dimX + (x + 1)] =
          X[y * dimX + x] +
          dt * rhs[y](X1[0 * dimX + x], X1[dimX + x], X1[2 * dimX + x]);
    }
  }
}

void rungeKutta4Order(
    std::vector<double> &X,
    std::vector<std::function<double(double, double, double)>> &rhs, double dt,
    std::size_t dimX, std::size_t dimY) {

  std::array<double, 3> X1{};
  std::array<double, 3> X2{};
  std::array<double, 3> X3{};
  std::array<double, 3> X4{};
  for (std::size_t x{0}; x < dimX - 1; ++x) {
    for (std::size_t y{0}; y < dimY - 1; ++y) {
      X1[y] = rhs[y](X[0 * dimX + x], X[dimX + x], X[2 * dimX + x]);
    } // f1
    for (std::size_t y{0}; y < dimY - 1; ++y) {
      X2[y] =
          rhs[y](X[0 * dimX + x] + dt / 2 * X1[0], X[dimX + x] + dt / 2 * X1[1],
                 X[2 * dimX + x] + dt / 2 * X1[2]);
    } // f2
    for (std::size_t y{0}; y < dimY - 1; ++y) {
      X3[y] =
          rhs[y](X[0 * dimX + x] + dt / 2 * X2[0], X[dimX + x] + dt / 2 * X2[1],
                 X[2 * dimX + x] + dt / 2 * X2[2]);
    } // f3
    for (std::size_t y{0}; y < dimY - 1; ++y) {
      X4[y] =
          rhs[y](X[0 * dimX + x] + dt / 2 * X3[0], X[dimX + x] + dt / 2 * X3[1],
                 X[2 * dimX + x] + dt / 2 * X3[2]);
    } // f4

    for (std::size_t y{0}; y < dimY - 1; ++y) {
      X[y * dimX + (x + 1)] =
          X[y * dimX + x] + dt / 6 * (X1[y] + 2 * X2[y] + 2 * X3[y] + +X4[y]);
    } // averaging
  }
}

int main() {
  Timer timeCpu;

  // solver piece

  constexpr double dt{0.001};
  constexpr double totalT{1.0};

  constexpr std::size_t dimX{static_cast<std::size_t>(totalT / dt)};
  constexpr std::size_t dimY{3};
  constexpr int particleNumber{1000000};
  std::vector<std::vector<double>> positionArray;

  std::vector<std::function<double(double, double, double)>> rhs;

  // functions  to be integrated (rhs)
  double sigma{10};
  double beta{8.0 / 3.0};
  double rho{28};

  rhs.push_back(
      [sigma](double x, double y, double z) { return sigma * (y - x); });
  rhs.push_back(
      [rho](double x, double y, double z) { return x * (rho - z) - y; });
  rhs.push_back(
      [beta](double x, double y, double z) { return x * y - beta * z; });

  // create X,Y,Z and set initial value
  for (int i{0}; i < particleNumber; ++i) {
    std::vector<double> X(dimX * dimY, 0.0);
    double x0{Random::get(1, 200) * 0.1};
    double y0{Random::get(1, 200) * 0.1};
    double z0{Random::get(1, 200) * 0.1};
    X[0] = x0;        // x initial
    X[dimX] = y0;     // y initial
    X[2 * dimX] = z0; // z initial
    positionArray.push_back(X);
  }

  // Launch threads
  std::vector<std::thread> threads;
  for (int i{0}; i < particleNumber; ++i) {
    threads.push_back(std::thread(rungeKutta4Order,
                                  std::ref(positionArray.data()[i]),
                                  std::ref(rhs), dt, dimX, dimY));
  }
  // Join threads before program execution terminates
  for (auto &th : threads) {
    th.join();
  }

  std::cout << timeCpu.elapsed() << " seconds elapsed." << '\n';

  return 0;
}

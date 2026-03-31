#include "helper.h"
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <vector>

double leftRectangleArea(std::vector<double> &xk, double dx, int N) {
  double area{0.0f};
  for (int i{0}; i < N; ++i) {
    area = area + sin(xk.data()[i]) * dx;
  }
  return area;
}

double rightRectangleArea(std::vector<double> &xk, double dx, int N) {
  double area{0.0f};
  for (int i{1}; i <= N; ++i) {
    area = area + sin(xk.data()[i]) * dx;
  }
  return area;
}

double trapezoidArea(std::vector<double> &xk, double dx, int N) {
  double area{0.0f};
  for (int i{0}; i < N; ++i) {
    area = area + dx / 2 * (sin(xk.data()[i]) + sin(xk.data()[i + 1]));
  }
  return area;
}

void printArray(std::vector<double> const &arr) {
  for (double const &item : arr) {
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

// vectorAdd addition
std::vector<double> vecAddCpu(std::vector<double> const &x,
                              std::vector<double> const &y) {

  std::size_t N{x.size()};
  std::vector<double> z(N, 0.0);
  for (std::size_t i{0}; i < N; ++i)
    z[i] = x[i] + y[i];
  return z;
}

std::vector<double> feSolver(std::vector<double> &A, std::vector<double> &x0,
                             double dt, int N) {
  std::vector<double> I{1, 0, 0, 1}; // identity
  std::vector<double> X(static_cast<std::size_t>(N), 0.0);
  X[0] = x0[0]; // initial position
  X[1] = x0[1]; // initial velocity
  std::vector<double> propagator{vecAddCpu(I, vectorScale(A, dt))}; // (I+A dt)

  for (int k{2}; k + 1 < N; k += 2) {
    std::vector<double> Xdot{
        mmCpu(propagator, {X.data()[k - 2], X.data()[k - 1]}, 2, 1, 2)};
    // printArray(Xdot);
    X.data()[k] = Xdot[0];
    X.data()[k + 1] = Xdot[1];
  }
  return X;
}

std::vector<double> verlet(std::vector<double> &x0, double dt, int N) {
  std::vector<double> X(static_cast<std::size_t>(N), 0.0);
  X[0] = x0[0]; // initial position
  X[1] = x0[0] + x0[1] * dt + 1.0f / 2.0f * (-1.0f / (x0[0] * x0[0])) * dt * dt;

  for (int k{1}; k < N; ++k) {
    X.data()[k + 1] = 2 * X.data()[k] - X.data()[k - 1] +
                      dt * dt * (-1.0f / (X.data()[k] * X.data()[k]));
  }
  return X;
}

template <typename T>
void derivativeForward(std::function<T(T)> &fn, std::vector<T> &array, T dt){
std::cout << dt << '\n';}

int main() {
  Timer t;
  double w{2 * M_PI}; // natural frequency
  double zeta{0.25};  // damping ratio

  int Nx{2};
  int Ny{2};
  std::vector<double> A{0, 1, -w * w, -2 * zeta * w}; // A vector
  std::vector<double> I{1, 0, 0, 1};                  // identity
  std::vector<double> x0{1, 1};                       // initial condition
  double dt{0.001};                                   // time step
  double T{10.0};                                     // total time

  int N{static_cast<int>(T / dt)};

  std::vector<double> scaledVec{vectorScale(A, dt)};
  std::vector<double> C(2, 0);
  std::vector<double> feResult{feSolver(A, x0, dt, N)};
  std::vector<double> verletResult{verlet(x0, dt, N)};
  // printArray(verletResult);

  std::cout << "total time is " << t.elapsed() << '\n';
  return 0;
}

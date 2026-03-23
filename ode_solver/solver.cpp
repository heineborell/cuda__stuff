#include "helper.h"
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numbers>
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

void printArray(std::vector<double> &arr) {
  for (double &item : arr) {
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
void mm(std::vector<double> &A, std::vector<double> &B, std::vector<double> &C,
        int Nay, int Nbx) {
  for (int y{0}; y < Nay; ++y) {
    for (int x{0}; x < Nbx; ++x) {
      double sum{0.0f};
      for (int i{0}; i < Nay; ++i) {
        sum += A.data()[y * Nay + i] * B.data()[i * Nay + x];
      }
      C.data()[y * Nay + x] = sum;
    }
  }
}

int main() {
  Timer t;
  double w{2 * std::numbers::pi}; // natural frequency
  double zeta{0.25};              // damping ratio

  int Nx{2};
  int Ny{2};
  std::vector<double> A{0, 1, -w * w, -2 * zeta * w}; // A vector
  std::vector<double> I{1, 0, 0, 1};                  // identity
  double dt{0.01};                                    // time step
  double T{10.0};                                     // total time

  std::vector<double> x0{2, 0}; // initial condition
  int N{static_cast<int>(T / dt)};

  // std::cout << "number of time intervals " << N << '\n';
  std::vector<double> scaledVec{vectorScale(A, dt)};
  std::vector<double> C(2, 0);
  mm_og(I, x0, C, 2, 1);

  // mm_cpu(I, x0, C, 2, 2, 0);

  printArray(C);
  // printArray(A);
  // printArray(scaledVec);

  std::cout << "total time is " << t.elapsed() << '\n';
  return 0;
}

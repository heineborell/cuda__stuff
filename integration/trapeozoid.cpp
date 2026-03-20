#include "helper.h"
#include <cmath>
#include <cstddef>
#include <iomanip>
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

void printArray(std::vector<double> &arr) {
  for (double &item : arr) {
    std::cout << item << '\n';
  }
}
std::vector<int>(32);

int main() {
  Timer t;
  int N{10000};
  double a{0.0f};
  double b{10.0f};
  double interval{b - a};
  double dx{interval / N};
  std::vector<double> xk(static_cast<std::size_t>(N + 1));

  // fill in the vector
  for (int i{0}; i <= N; ++i) {
    if (i == 0)
      xk.data()[i] = a;
    else
      xk.data()[i] = xk.data()[i - 1] + dx;
  }

  // printArray(xk);
  std::cout << std::setprecision(17);
  std::cout << " dx is " << dx << '\n';
  std::cout << " the exact value " << -(cos(b) - cos(a)) << '\n';
  std::cout << " left RectangleArea " << leftRectangleArea(xk, dx, N) << '\n';
  std::cout << " right RectangleArea " << rightRectangleArea(xk, dx, N) << '\n';
  std::cout << " trapezoid area " << trapezoidArea(xk, dx, N) << '\n';
  std::cout << "total time is " << t.elapsed() << '\n';
  return 0;
}

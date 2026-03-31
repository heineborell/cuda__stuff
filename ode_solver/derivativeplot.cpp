#include "math.h"
#include "raylib.h"
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

const int screenWidth{1960};
const int screenHeight{1200};

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
  df[0] = (f(timeArray[1]) - f(timeArray[0])) / dt;
  for (std::size_t i{1}; i < N; ++i) {
    df[i] = (f(timeArray[i + 1]) - f(timeArray[i - 1])) / (2 * dt);
  }
}

void plotter(int dimX, std::vector<float> &resultFloatx,
             std::vector<float> &resultFloaty, float xRange, Color color) {
  for (std::size_t i{0}; i < dimX - 1; ++i) {
    float x1{resultFloatx[i]};     // current x value
    float x2{resultFloatx[i + 1]}; // next x value
    float y1{resultFloaty[i]};
    float y2{resultFloaty[i + 1]};

    // Map x and y values to screen coordinates
    Vector2 start = {screenWidth / 2 + x1 * (screenWidth / (2 * xRange)),
                     screenHeight / 2 - y1 * (screenHeight / (2 * xRange))};
    Vector2 end = {screenWidth / 2 + x2 * (screenWidth / (2 * xRange)),
                   screenHeight / 2 - y2 * (screenHeight / (2 * xRange))};
    DrawLineEx(start, end, 1.5f, color);
  }
}
int main() {
  using Real = float;
  Real dt{1E-1}; // time step
  Real T{10};    // total time

  // function
  std::function<Real(Real)> fn{[](Real t) { return sin(t); }};
  std::function<Real(Real)> dfn{[](Real t) { return cos(t); }};
  int N{static_cast<int>(T / dt)};

  // define the derivative vectors
  std::vector<Real> time(N, 0.0);
  std::vector<Real> dfForward(N, 0.0);
  std::vector<Real> dfCenter(N, 0.0);
  std::vector<Real> dfExact(N, 0.0);

  // populate time array
  timeVec(time, dt);

  // calculate derivatives
  derivativeForward(dfForward, fn, time, dt);
  derivativeCenter(dfCenter, fn, time, dt);
  evaluate(dfExact, dfn, time);

  // printArray(df);
  std::vector<Real> errorForward{vecSubtractCpu(dfExact, dfForward)};
  std::vector<Real> errorCenter{vecSubtractCpu(dfExact, dfCenter)};
  std::cout << std::abs(errorForward.back()) << " Forward Derivative error "
            << '\n';
  std::cout << std::abs(errorCenter.back()) << " Forward Center error " << '\n';
  // printArray(time);

  // plotting range
  float xRange{10.0f};

  InitWindow(screenWidth, screenHeight, "X-Y plot");
  SetTargetFPS(60);
  const float zoomSpeed{1.1f};

  while (!WindowShouldClose()) {
    if (IsKeyPressed(KEY_UP))
      xRange /= zoomSpeed; // zoom in
    if (IsKeyPressed(KEY_DOWN))
      xRange *= zoomSpeed; // zoom in
    if (xRange < 0.1f)
      xRange = 0.1f;
    if (xRange > 50.0f)
      xRange = 50.0f;

    int dimX{10};
    float step{xRange / dimX}; // step size for plotting

    BeginDrawing();
    ClearBackground(BLACK);
    DrawFPS(10, 10);
    DrawLine(screenWidth / 2, 0, screenWidth / 2, screenHeight, GRAY);
    DrawLine(0, screenHeight / 2, screenWidth, screenHeight / 2, GRAY);

    DrawText("Y", screenWidth / 2 + 5, 5, 20, GRAY);
    DrawText("X", screenWidth - 20, screenHeight / 2 + 5, 20, GRAY);
    plotter(N, time, dfExact, xRange, GREEN);
    plotter(N, time, dfCenter, xRange, BLUE);
    plotter(N, time, dfForward, xRange, RED);

    EndDrawing();
  }

  CloseWindow();
  return 0;
}

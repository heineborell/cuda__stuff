#include "math.h"
#include "raylib.h"
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numbers>
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

int main() {
  // solver piece
  double w{2 * M_PI}; // natural frequency
  double zeta{0.25};  // damping ratio

  int Nx{2};
  int Ny{2};
  std::vector<double> A{0, 1, -w * w, -2 * zeta * w}; // A vector
  std::vector<double> I{1, 0, 0, 1};                  // identity
  std::vector<double> x0{2, 0};                       // initial condition
  double dt{0.01};                                    // time step
  float xRange{
      10.0f}; // x will range from -4 to 4 but then changed by scrolling

  int N{static_cast<int>(xRange / dt)};
  std::vector<double> result{feSolver(A, x0, dt, N)};
  std::vector<float> resultFloat{castArray(result)};

  const int screenWidth{1000};
  const int screenHeight{640};
  InitWindow(screenWidth, screenHeight, "polynomialWave");
  SetTargetFPS(60);
  const int wavePoints{1000};
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

    float step{xRange / wavePoints}; // step size for plotting

    BeginDrawing();
    ClearBackground(BLACK);

    DrawLine(screenWidth / 2, 0, screenWidth / 2, screenHeight, GRAY);
    DrawLine(0, screenHeight / 2, screenWidth, screenHeight / 2, GRAY);

    DrawText("Y", screenWidth / 2 + 5, 5, 20, GRAY);
    DrawText("X", screenWidth - 20, screenHeight / 2 + 5, 20, GRAY);
    for (std::size_t i{0}; i < N; ++i) {
      float x1{0 + i * step}; // current x value
      float x2{x1 + step};    // next x value
      float y1{resultFloat.data()[i]};
      float y2{resultFloat.data()[i + 2]};

      // Map x and y values to screen coordinates
      Vector2 start = {screenWidth / 2 + x1 * (screenWidth / (2 * xRange)),
                       screenHeight / 2 - y1 * (screenHeight / (2 * xRange))};
      Vector2 end = {screenWidth / 2 + x2 * (screenWidth / (2 * xRange)),
                     screenHeight / 2 - y2 * (screenHeight / (2 * xRange))};
      if (i % 2 == 0)
        DrawLineEx(start, end, 1.0f, GREEN);
      else
        DrawLineEx(start, end, 1.0f, MAROON);
    }

    EndDrawing();
  }

  CloseWindow();
  return 0;
}

#include "math.h"
#include "raylib.h"
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
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

  std::vector<double> X1(dimX * dimY, 0.0);
  std::vector<double> X2(dimX * dimY, 0.0);
  std::vector<double> X3(dimX * dimY, 0.0);
  std::vector<double> X4(dimX * dimY, 0.0);
  for (std::size_t x{0}; x < dimX - 1; ++x) {
    for (std::size_t y{0}; y < dimY - 1; ++y) {
      X1[y * dimX + (x + 1)] =
          X[y * dimX + x] + dt / 2 *
                                rhs[y](X[0 * dimX + x], X[dimX + x],
                                       X[2 * dimX + x]); // f1

      X2[y * dimX + (x + 1)] =
          X[y * dimX + x] + dt / 2 *
                                rhs[y](X1[0 * dimX + x], X1[dimX + x],
                                       X1[2 * dimX + x]); // f2
      X3[y * dimX + (x + 1)] =
          X[y * dimX + x] + dt / 2 *
                                rhs[y](X2[0 * dimX + x], X2[dimX + x],
                                       X2[2 * dimX + x]); // f3
      X4[y * dimX + (x + 1)] =
          X[y * dimX + x] + dt / 2 *
                                rhs[y](X3[0 * dimX + x], X3[dimX + x],
                                       X3[2 * dimX + x]); // half-step
      //
      X[y * dimX + (x + 1)] =
          X[y * dimX + x] +
          dt / 6 *
              (X1[y * dimX + x] + 2 * X2[y * dimX + x] + 2 * X3[y * dimX + x] +
               +X4[y * dimX + x]); // averaging
    }
  }
}

int main() {

  // solver piece

  constexpr double dt{0.01};
  constexpr double totalT{150.0};
  float xRange{4.0f};

  constexpr std::size_t dimX{static_cast<std::size_t>(totalT / dt)};
  // const int dimX{5};
  constexpr std::size_t dimY{4};

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

  std::vector<double> X(dimX * dimY, 0.0);
  X[0] = 1.0;       // x initial
  X[dimX] = 1.0;    // y initial
  X[2 * dimX] = 27; // z initial

  std::vector<double> X2(dimX * dimY, 0.0);
  X2[0] = 1.0;       // x initial
  X2[dimX] = 1.0;    // y initial
  X2[2 * dimX] = 27; // z initial

  std::vector<double> X4(dimX * dimY, 0.0);
  X4[0] = 1.0;       // x initial
  X4[dimX] = 1.0;    // y initial
  X4[2 * dimX] = 27; // z initial

  // time vector tk
  timeVec(X, dimX, dimY, dt);

  // integrator
  forwardEuler(X, rhs, dt, dimX, dimY);
  rungeKutta2Order(X2, rhs, dt, dimX, dimY);
  rungeKutta4Order(X4, rhs, dt, dimX, dimY);
  std::vector<float> resultFloat{castArray(X)};
  std::vector<float> resultFloat2{castArray(X2)};
  std::vector<float> resultFloat4{castArray(X4)};

  const int screenWidth{1960};
  const int screenHeight{1200};
  InitWindow(screenWidth, screenHeight, "runge kutta 2");
  SetTargetFPS(60);
  const float zoomSpeed{1.1f};

  // Define the camera to look into our 3d world
  Camera3D camera = {0};
  camera.position = (Vector3){10.0f, 10.0f, 10.0f}; // Camera position
  camera.target = (Vector3){0.0f, 0.0f, 0.0f};      // Camera looking at point
  camera.up =
      (Vector3){0.0f, 1.0f, 0.0f}; // Camera up vector (rotation towards target)
  camera.fovy = 45.0f;             // Camera field-of-view Y
  camera.projection = CAMERA_PERSPECTIVE;
  DisableCursor();

  while (!WindowShouldClose()) {
    // Update
    //----------------------------------------------------------------------------------
    UpdateCamera(&camera, CAMERA_FREE);

    if (IsKeyPressed(KEY_Z))
      camera.target = (Vector3){0.0f, 0.0f, 0.0f};
    //----------------------------------------------------------------------------------

    BeginDrawing();
    ClearBackground(BLACK);
    BeginMode3D(camera);
    // Note that in raylib the up coordinate is y

    for (std::size_t i{0}; i < dimX - 1; ++i) {
      float x1{resultFloat[i]};     // current x value
      float x2{resultFloat[i + 1]}; // next x value
      float y1{resultFloat[dimX + i]};
      float y2{resultFloat[dimX + i + 1]};
      float z1{resultFloat[2 * dimX + i]};
      float z2{resultFloat[2 * dimX + +i + 1]};
      DrawLine3D({x1, y1, z1}, {x2, y2, z2}, GREEN);
    }
    for (std::size_t i{0}; i < dimX - 1; ++i) {
      float x1{resultFloat2[i]};     // current x value
      float x2{resultFloat2[i + 1]}; // next x value
      float y1{resultFloat2[dimX + i]};
      float y2{resultFloat2[dimX + i + 1]};
      float z1{resultFloat2[2 * dimX + i]};
      float z2{resultFloat2[2 * dimX + +i + 1]};
      DrawLine3D({x1, y1, z1}, {x2, y2, z2}, BLUE);
    }
    for (std::size_t i{0}; i < dimX - 1; ++i) {
      float x1{resultFloat4[i]};     // current x value
      float x2{resultFloat4[i + 1]}; // next x value
      float y1{resultFloat4[dimX + i]};
      float y2{resultFloat4[dimX + i + 1]};
      float z1{resultFloat4[2 * dimX + i]};
      float z2{resultFloat4[2 * dimX + +i + 1]};
      DrawLine3D({x1, y1, z1}, {x2, y2, z2}, RED);
    }
    EndMode3D();
    DrawFPS(10, 10);
    EndDrawing();
  }

  CloseWindow();
  return 0;
}

#include "math.h"
#include "raylib.h"
#include <cstddef>
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


std::vector<double> timeVec(int N, double dt){
       std::vector<double> Tk(N,0.0);
       for(int i{0}; i< N;++i)
        Tk.data()[i+1]=Tk.data()[i]+dt;
       return Tk;}


template <typename Func>
std::vector<double> rk2firstOrder(std::vector<double> &X,const Func &fn, double dt, int N) {
  for (int k{0}; k < N; ++k) {
    X.data()[k + 1] = X.data()[k] +dt ;
  }

return X;}

int main() {
  // solver piece

  double dt{0.001};                                   
  float xRange{2.5f};

  int N{static_cast<int>(xRange / dt)};
  std::vector<double> X(static_cast<std::size_t>(N), 0.0);
  X.data()[0]= 1.0;

  std::vector<double> tk {timeVec(N, dt)};
  auto rhs=[](double x){return (-1.0 / (x * x));};
  std::vector<double> result{rk2firstOrder(X, rhs, dt, N)};
  std::vector<float> resultFloat{castArray(result)};

  const int screenWidth{1960};
  const int screenHeight{800};
  InitWindow(screenWidth, screenHeight, "polynomialWave");
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

    float step{xRange / N}; // step size for plotting

    BeginDrawing();
    ClearBackground(BLACK);

    DrawLine(screenWidth / 2, 0, screenWidth / 2, screenHeight, GRAY);
    DrawLine(0, screenHeight / 2, screenWidth, screenHeight / 2, GRAY);

    DrawText("Y", screenWidth / 2 + 5, 5, 20, GRAY);
    DrawText("T", screenWidth - 20, screenHeight / 2 + 5, 20, GRAY);
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
      DrawLineEx(start, end, 1.5f, GREEN);
    }

    EndDrawing();
  }

  CloseWindow();
  return 0;
}

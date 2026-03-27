#include "math.h"
#include "raylib.h"
#include <cstddef>
#include <iostream>
#include <vector>
#include <iomanip>

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


void timeVec(std::vector<double>&vec,int gridDimX,int gridDimY, double dt){
  int N {static_cast<int>(vec.size())};
       for(int i{gridDimX*(gridDimY-1)}; i< N;++i)
        vec.data()[i+1]=vec.data()[i]+dt;}  

void showMatrix(std::vector<double> &vec,int gridDimX, int gridDimY) {
  int N {static_cast<int>(vec.size())};
  std::cout << std::setprecision(3);
  for(int i {0}; i<N;++i){
    if(i%gridDimX==0)
      std::cout << '\n';
    std::cout << vec[i] << " ";}
  }




void rk2firstOrder(std::vector<double> &X,double dt, int dimX, int dimY) {
  //functions  to be integrated (rhs)
  double sigma {10};
  double rho {8.0/3.0};
  double beta {28};

  auto fnX=[sigma](double x,double y,double z ){return sigma*(y-x);};
  auto fnY=[rho](double x,double y,double z ){return x*(rho-z)-y;};
  auto fnZ=[beta](double x,double y,double z ){return x*y-beta*z;};

    for(int x{0};x < dimX-1;++x)
      for(int y{0};y < dimY-1;++y)
      X.data()[y*dimX+(x+1)]=X.data()[y*dimX+x]+dt*fnX(X.data()[y*dimX+x],X.data()[(y+1)*dimX+x],X.data()[(y+2)*dimX+x]);
 }

int main() {
  // solver piece

  double dt{0.01};                                   
  double totalT{4.0};
  float xRange{4.0f};

  const int dimX{static_cast<int>(totalT/ dt)};
  // const int dimX{5};
  constexpr int dimY {4};
  
  // create X,Y,Z and set initial value

  std::vector<double> X(static_cast<std::size_t>(dimX*dimY), 0.0);
  X.data()[0]= -8.0; //x initial
  X.data()[dimX]= 8.0; // y initial
  X.data()[2*dimX]= 27; // z initial 

  // time vector tk
  timeVec(X,dimX,dimY, dt);

  //integrator
  showMatrix(X, dimX, dimY);
  std::cout << "the processed matrix" << '\n';
  rk2firstOrder(X, dt, dimX,dimY);
  std::vector<float> resultFloat{castArray(X)};
  showMatrix(X, dimX, dimY);

  const int screenWidth{1960};
  const int screenHeight{800};
  InitWindow(screenWidth, screenHeight, "runge kutta 2");
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

    float step{xRange / dimX}; // step size for plotting

    BeginDrawing();
    ClearBackground(BLACK);

    DrawLine(screenWidth / 2, 0, screenWidth / 2, screenHeight, GRAY);
    DrawLine(0, screenHeight / 2, screenWidth, screenHeight / 2, GRAY);

    DrawText("Y", screenWidth / 2 + 5, 5, 20, GRAY);
    DrawText("T", screenWidth - 20, screenHeight / 2 + 5, 20, GRAY);
    for (std::size_t i{0}; i < dimX-1; ++i) {
      float x1{resultFloat.data()[i]}; // current x value
      float x2{resultFloat.data()[i+1]};    // next x value
      float y1{resultFloat.data()[dimX+i]};
      float y2{resultFloat.data()[dimX+i+1]};

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

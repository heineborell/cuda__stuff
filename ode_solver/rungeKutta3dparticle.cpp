#include "math.h"
#include "raylib.h"
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>
#include "raymath.h"

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

void rk2firstOrder(
    std::vector<double> &X,
    std::vector<std::function<double(double, double, double)>> &rhs, double dt,
    int dimX, int dimY) {

  for (int x{0}; x < dimX - 1; ++x) {
    for (int y{0}; y < dimY - 1; ++y) {
      X.data()[y * dimX + (x + 1)] =
          X.data()[y * dimX + x] + dt * rhs.data()[y](X.data()[0 * dimX + x],
                                                      X.data()[dimX + x],
                                                      X.data()[2 * dimX + x]);
    }
  }
}

int main() {

  // solver piece

  double dt{0.01};
  double totalT{150.0};
  float xRange{4.0f};

  const int dimX{static_cast<int>(totalT / dt)};
  // const int dimX{5};
  constexpr int dimY{4};

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

  std::vector<double> X(static_cast<std::size_t>(dimX * dimY), 0.0);
  X.data()[0] = 1.0;       // x initial
  X.data()[dimX] = 1.0;    // y initial
  X.data()[2 * dimX] = 27; // z initial

  // time vector tk
  timeVec(X, dimX, dimY, dt);

  // integrator
  showMatrix(X, dimX, dimY);
  std::cout << "the processed matrix" << '\n';
  rk2firstOrder(X, rhs, dt, dimX, dimY);
  std::vector<float> resultFloat{castArray(X)};
  showMatrix(X, dimX, dimY);

  const int screenWidth{1960};
  const int screenHeight{1200};
  InitWindow(screenWidth, screenHeight, "runge kutta 2");
  SetTargetFPS(60);
  const float zoomSpeed{1.1f};

Camera3D camera = { 0 };
    camera.position = (Vector3){ 80.0f, 80.0f, 80.0f };
    camera.target = (Vector3){ 0.0f, 0.0f, 0.0f };
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
	
    const int count = 5000;

    // 1. Setup the Mesh and Material
    Mesh sphereMesh = GenMeshSphere(0.3f, 16, 16);
    Model sphereModel = LoadModelFromMesh(sphereMesh);


// 1. Load the simple instancing shader
// We use NULL for the fragment shader to use raylib's default "unlit" look
Shader shader = LoadShader("shaders/base_instancing.vs", NULL);

// 2. Link the instance attribute
shader.locs[SHADER_LOC_MATRIX_MVP] = GetShaderLocation(shader, "mvp");
shader.locs[SHADER_LOC_MATRIX_MODEL] = GetShaderLocationAttrib(shader, "instanceTransform");

// 3. Assign to model
sphereModel.materials[0].shader = shader;
sphereModel.materials[0].maps[MATERIAL_MAP_DIFFUSE].color = GREEN; // All spheres will be green or whatever you choose
  
    // 2. Prepare the Transformation Matrices
    std::vector<Matrix> transforms(count);
    for (int i = 0; i < count; i++) {
        // Create a random position for each sphere
        Vector3 pos = { 
            (float)GetRandomValue(-50, 50), 
            (float)GetRandomValue(-50, 50), 
            (float)GetRandomValue(-50, 50) 
        };
        transforms[i] = MatrixTranslate(pos.x, pos.y, pos.z);
    }

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

// Optional: Move spheres every frame
    for (int i = 0; i < count; i++) {
        // Example: Subtle floating movement
        float time = (float)GetFrameTime();
        std::cout << time <<'\n';
        transforms[i] = MatrixMultiply(transforms[i], MatrixTranslate(0, sinf(time + i) * 0.01f, 0));
    }
    DrawMeshInstanced(sphereMesh, sphereModel.materials[0], transforms.data(), count);


    EndMode3D();
    DrawFPS(10, 10);
    EndDrawing();
  }

  CloseWindow();
  return 0;
}

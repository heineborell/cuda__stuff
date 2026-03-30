#include "math.h"
#include "raylib.h"
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>
#include "raymath.h"


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
  // DisableCursor();
	
    const int count = 70000;

    // 1. Setup the Mesh and Material
    Mesh sphereMesh = GenMeshSphere(0.05f, 16, 16);
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

  auto test {[](float time) { return time; }};
  while (!WindowShouldClose()) {
    // Update
    //----------------------------------------------------------------------------------
    UpdateCamera(&camera, CAMERA_THIRD_PERSON);

    if (IsKeyPressed(KEY_Z))
      camera.target = (Vector3){0.0f, 0.0f, 0.0f};
    //----------------------------------------------------------------------------------


    BeginDrawing();
    ClearBackground(BLACK);
    BeginMode3D(camera);

// Optional: Move spheres every frame
    for (int i = 0; i < count; i++) {
        // float time = (float)GetFrameTime();
        float time = 0.01; 
        float x {transforms[i].m12};
        float y {transforms[i].m13};
        float z {transforms[i].m14};
        transforms[i] = MatrixMultiply(transforms[i], MatrixTranslate(time*rhs[0](x,y,z),time*rhs[1](x,y,z), time*rhs[2](x,y,z)));
    }
    DrawMeshInstanced(sphereMesh, sphereModel.materials[0], transforms.data(), count);


    EndMode3D();
    DrawFPS(10, 10);
    EndDrawing();
  }

  CloseWindow();
  return 0;
}

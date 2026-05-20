#include "Random.h"
#include "helper.h"
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

constexpr std::size_t particleNumber{300000};
constexpr double dt{0.001};
constexpr double totalT{1.0};
constexpr std::size_t dimT{static_cast<std::size_t>(totalT / dt)};
constexpr std::size_t dimY{3};
constexpr std::size_t arraySize{particleNumber * dimT * dimY};

void printArray(std::vector<double> const &arr) {
  for (double const &item : arr) {
    std::cout << item << '\n';
  }
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

__device__ double fx(double x, double y, double z, double param1) {
  return param1 * (y - x);
}

__device__ double fy(double x, double y, double z, double param1) {
  return x * (param1 - z) - y;
}

__device__ double fz(double x, double y, double z, double param1) {
  return x * y - param1 * z;
}

__global__ void rungeKutta4OrderGpu(double *X, double dt) {

  unsigned int i{blockDim.x * blockIdx.x + threadIdx.x};
  if (i < particleNumber) {

    double X1[3];
    double X2[3];
    double X3[3];
    double X4[3];

    for (std::size_t t{0}; t < dimT - 1; ++t) {

      const double sigma{10};
      const double beta{8.0 / 3.0};
      const double rho{28};

      const std::size_t offset = t * particleNumber + i;
      const std::size_t stride = dimT * particleNumber;

      const double valX = X[0 * stride + offset];
      const double valY = X[1 * stride + offset];
      const double valZ = X[2 * stride + offset];

      X1[0] = fx(valX, valY, valZ, sigma); // f1
      X1[1] = fy(valX, valY, valZ, rho);
      X1[2] = fz(valX, valY, valZ, beta);

      X2[0] = fx(valX + dt / 2 * X1[0], valY + dt / 2 * X1[1], // f2
                 valZ + dt / 2 * X1[2], sigma);
      X2[1] = fy(valX + dt / 2 * X1[0], valY + dt / 2 * X1[1],
                 valZ + dt / 2 * X1[2], rho);
      X2[2] = fz(valX + dt / 2 * X1[0], valY + dt / 2 * X1[1],
                 valZ + dt / 2 * X1[2], beta);

      X3[0] = fx(valX + dt / 2 * X2[0], valY + dt / 2 * X2[1], // f3
                 valZ + dt / 2 * X2[2], sigma);
      X3[1] = fy(valX + dt / 2 * X2[0], valY + dt / 2 * X2[1],
                 valZ + dt / 2 * X2[2], rho);
      X3[2] = fz(valX + dt / 2 * X2[0], valY + dt / 2 * X2[1],
                 valZ + dt / 2 * X2[2], beta);

      X4[0] = fx(valX + dt / 2 * X3[0], valY + dt / 2 * X3[1], // f4
                 valZ + dt / 2 * X3[2], sigma);
      X4[1] = fy(valX + dt / 2 * X3[0], valY + dt / 2 * X3[1],
                 valZ + dt / 2 * X3[2], rho);
      X4[2] = fz(valX + dt / 2 * X3[0], valY + dt / 2 * X3[1],
                 valZ + dt / 2 * X3[2], beta);

      for (std::size_t y{0}; y < dimY; ++y) {
        X[y * stride + (particleNumber * (t + 1) + i)] =
            X[y * stride + offset] +
            dt / 6 * (X1[y] + 2 * X2[y] + 2 * X3[y] + +X4[y]);
      } // averaging
    }
  }
}

void rungeKuttaGPU(std::vector<double> &X, double dt) {

  double *X_d;

  Timer t_allocategpu;
  cudaMalloc((void **)(&X_d), arraySize * sizeof(double));
  cudaDeviceSynchronize();
  std::cout << " Memories allocated on GPU! " << t_allocategpu.elapsed()
            << " secs." << '\n';

  // Copy data to the GPU.

  Timer t_copygpu;
  cudaMemcpy(X_d, X.data(), arraySize * sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  std::cout << " Data copied to GPU! " << t_copygpu.elapsed() << " secs."
            << '\n';

  const unsigned int numThreadsPerBlock{512}; // again multiple of 32.
  const unsigned int numBlocks =
      (arraySize + numThreadsPerBlock - 1) /
      numThreadsPerBlock; // since I have 512 threads per block and also N
                          // threads the N/512 will be number of blocks. also
                          // use a trick for the ceiling and launch extra
                          // threads!

  Timer t_kernelgpu;
  // Your Kernel Call Here
  rungeKutta4OrderGpu<<<numBlocks, numThreadsPerBlock>>>(
      X_d, dt); // provide the configuration inside <<< >>>>. Now each thread
                // will execute this function! (so no for loop or sth)

  cudaDeviceSynchronize(); // wait for GPU to finish!
  std::cout << "GPU computation time " << t_kernelgpu.elapsed() << " secs."
            << '\n';
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
  }
  // Copy from the GPU
  cudaMemcpy(X.data(), X_d, arraySize * sizeof(double), cudaMemcpyDeviceToHost);

  // Deallocate memory on the GPU
  cudaFree(X_d);
}

int main() {

  std::vector<double> X(arraySize, 0.0);

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
  for (int i{0}; i < particleNumber; ++i) {
    double x0{Random::get(1, 200) * 0.1};
    double y0{Random::get(1, 200) * 0.1};
    double z0{Random::get(1, 200) * 0.1};
    X[0 * particleNumber * dimT + i] = x0; // x initial
    X[1 * particleNumber * dimT + i] = y0; // y initial
    X[2 * particleNumber * dimT + i] = z0; // z initial
  }
  // showMatrix(X, particleNumber * dimT, dimY);

  rungeKuttaGPU(X, dt);
  cudaDeviceSynchronize();

  // showMatrix(X, particleNumber * dimT, dimY);
  // showMatrix(Xcpu, particleNumber * dimT, dimY);

  return 0;
}

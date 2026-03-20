#include "helper.h"
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

double leftRectangleArea_cpu(std::vector<double> &xk, double dx, int N) {
  double area{0.0f};
  for (int i{0}; i < N; ++i) {
    area = area + sin(xk.data()[i]) * dx;
  }
  return area;
}

double rightRectangleArea_cpu(std::vector<double> &xk, double dx, int N) {
  double area{0.0f};
  for (int i{1}; i <= N; ++i) {
    area = area + sin(xk.data()[i]) * dx;
  }
  return area;
}

double trapezoidArea_cpu(std::vector<double> &xk, double dx, int N) {
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

__global__ void integrateKernel(double *xk_d, double *area_d, double *result_d,
                                double dx, int N) {
  unsigned int i{blockDim.x * blockIdx.x + threadIdx.x};
  if (i < N)
    area_d[i] = sin(xk_d[i]) * dx; // so this is wrong this introduces a race condition.
  for (unsigned j{0}; j < N; ++j) {
    *result_d += area_d[j]; 
  }
}

// __global__ void integrateKernel(double *xk_d,
//                                 double *result_d,
//                                 double dx,
//                                 int N)
// {
//     unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
//
//     if (i < N) {
//         double val = sin(xk_d[i]) * dx;
//         atomicAdd(result_d, val);
//     }
// }
//
// here each thread has its own val as variable because kernel is executed for
// eacch thread. then each thread reads xk_d from global memory. reads are not
// problem you can read from the global memory easily (sum is register but can
// spill to local mem). in the matrix example since row, acolumn is unqiue to
// each thread there is no race condition

void integrateGpu(std::vector<double> &xk, double *result, double dx, int N) {
  double *xk_d;
  double *area_d;
  double *result_d;

  Timer t_allocategpu;
  cudaMalloc((void **)&xk_d, (N + 1) * sizeof(double));
  cudaMalloc((void **)&area_d, (N + 1) * sizeof(double));
  cudaMalloc((void **)&result_d, sizeof(double));
  std::cout << " Memories allocated on GPU! " << t_allocategpu.elapsed()
            << " secs." << '\n';

  Timer t_copygpu;
  cudaMemcpy(xk_d, xk.data(), (N + 1) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(result_d, result, sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  std::cout << " Data copied to GPU! " << t_copygpu.elapsed() << " secs."
            << '\n';

  const unsigned int numThreadsPerBlock{32};
  const unsigned int numBlocks = ((N + 1) + 32 - 1) / 32;

  Timer t_kernelgpu;
  integrateKernel<<<numBlocks, numThreadsPerBlock>>>(xk_d, area_d, result_d, dx,
                                                     N);
  cudaDeviceSynchronize(); // wait for GPU to finish!
  std::cout << "GPU computation time " << t_kernelgpu.elapsed() << " secs."
            << '\n';

  cudaMemcpy(result, result_d, sizeof(double), cudaMemcpyDeviceToHost);

  Timer t_deallocategpu;
  cudaFree(xk_d);
  cudaFree(area_d);
  cudaFree(result_d);
  std::cout << " Memories freed for GPU! " << t_deallocategpu.elapsed() << '\n';
}

int main() {
  cudaDeviceSynchronize(); // wait for GPU to finish!

  int N{1000000};
  double a{0.0f};
  double b{10.0f};
  double interval{b - a};
  double dx{interval / N};
  double result{0};
  std::vector<double> xk(static_cast<std::size_t>(N + 1));

  // fill in the vector
  for (int i{0}; i <= N; ++i) {
    if (i == 0)
      xk.data()[i] = a;
    else
      xk.data()[i] = xk.data()[i - 1] + dx;
  }

  std::cout << std::setprecision(17);
  std::cout << xk.size() << '\n';
  // std::cout << " dx is " << dx << '\n';
  std::cout << " the exact value " << -(cos(b) - cos(a)) << '\n';
  Timer t;
  std::cout << " left RectangleArea " << leftRectangleArea_cpu(xk, dx, N)
            << '\n';
  std::cout << " right RectangleArea " << rightRectangleArea_cpu(xk, dx, N)
            << '\n';
  std::cout << " trapezoid area " << trapezoidArea_cpu(xk, dx, N) << '\n';
  std::cout << "total time is " << t.elapsed() << '\n';

  // Timer t_cpu;
  // rgb2gray_cpu(red.data(),green.data(),blue.data(),gray.data(),width,height);
  // std::cout << " Cpu elapsed time " << t_cpu.elapsed() << '\n';

  Timer t_totalgpu;
  integrateGpu(xk, &result, dx, N);
  cudaDeviceSynchronize();
  std::cout << "GPU result leftarea " << result << '\n';
  std::cout << " GPU elapsed time " << t_totalgpu.elapsed() << '\n';

  return 0;
}

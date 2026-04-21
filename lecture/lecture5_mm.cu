#include "Random.h"
#include "helper.h"
#include <iostream>
#include <vector>
#define TILE_DIM 32
void mm_cpu(std::vector<float> &A, std::vector<float> &B, std::vector<float> &C,
            int N) {
  for (int y{0}; y < N; ++y) {
    for (int x{0}; x < N; ++x) {
      float sum{0.0f};
      for (int i{0}; i < N; ++i) {
        sum += A.data()[y * N + i] * B.data()[i * N + x];
      }
      C.data()[y * N + x] = sum;
    }
  }
}

__global__ void mm_kernel(float *A, float *B, float *C, int N) {
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int column = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float A_s[TILE_DIM][TILE_DIM]; // shared memory arrays
  __shared__ float B_s[TILE_DIM][TILE_DIM];

  float sum = 0.0f;
  // loop over tiles
  for (unsigned int tile = 0; tile < N / TILE_DIM; ++tile) {
    A_s[threadIdx.y][threadIdx.x] = A[row * N + tile * TILE_DIM + threadIdx.x];
    B_s[threadIdx.y][threadIdx.x] =
        B[(tile * TILE_DIM + threadIdx.y) * N + column];
    __syncthreads();
    // then do the partial sums
    for (unsigned int i = 0; i < TILE_DIM; ++i)
      sum += A_s[threadIdx.y][i] * B_s[i][threadIdx.x];
    __syncthreads();
  }
  C[row * N + column] = sum;
}

void mm_gpu(std::vector<float> &A, std::vector<float> &B, std::vector<float> &C,
            int N) {

  float *A_d;
  float *B_d;
  float *C_d;

  Timer t_allocategpu;
  cudaMalloc((void **)(&A_d), N * N * sizeof(float));
  cudaMalloc((void **)(&B_d), N * N * sizeof(float));
  cudaMalloc((void **)(&C_d), N * N * sizeof(float));
  cudaDeviceSynchronize();
  std::cout << " Memories allocated on GPU! " << t_allocategpu.elapsed()
            << " secs." << '\n';

  // Copy data to the GPU.

  Timer t_copygpu;
  cudaMemcpy(A_d, A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(C_d, C.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  std::cout << " Data copied to GPU! " << t_copygpu.elapsed() << " secs."
            << '\n';

  // Call a GPU function (launch a grid of threads) gotta tell how many blocks
  // and how many threads for block. so this thing is interesting as we defined
  // the matrices as flattned arrays in the cpu side we push them as flattened
  // arrays to gpu but make the calculations as 2d objects.
  dim3 numThreadsPerBlock(32, 32); //(x,y,z)
  dim3 numBlocks((N + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x,
                 (N + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);

  Timer t_kernelgpu;
  mm_kernel<<<numBlocks, numThreadsPerBlock>>>(A_d, B_d, C_d, N);
  cudaDeviceSynchronize(); // wait for GPU to finish!
  std::cout << "GPU computation time " << t_kernelgpu.elapsed() << " secs."
            << '\n';

  // Copy from the GPU
  cudaMemcpy(C.data(), C_d, N * N, cudaMemcpyDeviceToHost);

  Timer t_deallocategpu;
  // Deallocate memory on the GPU
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
  cudaDeviceSynchronize();
  std::cout << " Memories freed for GPU! " << t_deallocategpu.elapsed() << '\n';
}

int main() {

  int N{1000};

  std::vector<float> A(N * N);
  std::vector<float> B(N * N);
  std::vector<float> C(N * N);

  for (int y{0}; y < N; ++y) {
    for (int x{0}; x < N; ++x) {
      A.data()[y * N + x] = 0.1 * Random::get(1, 100);
    }
  }
  for (int y{0}; y < N; ++y) {
    for (int x{0}; x < N; ++x) {
      B.data()[y * N + x] = 0.1 * Random::get(1, 100);
    }
  }
  std::cout << "matrix size is " << A.size() << '\n';

  Timer t_cpu;
  mm_cpu(A, B, C, N);
  std::cout << " Cpu elapsed time " << t_cpu.elapsed() << '\n';

  // Sample call: Random::get(1L, 6L);             // returns long
  Timer t_totalgpu;
  mm_gpu(A, B, C, N);
  cudaDeviceSynchronize();
  std::cout << " GPU elapsed time " << t_totalgpu.elapsed() << '\n';

  return 0;
}

#include <iostream>
#include "helper.h"
#include <vector>
#include <cstdlib>
#include <vector>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>
#include "Random.h"

void mm_cpu(std::vector<float> &A, std::vector<float> &B,std::vector<float> &C, int N){}

__global__ void mm_kernel(float* A,float* B, float* C,int N){
}

void mm_gpu(std::vector<float> &A, std::vector<float> &B,std::vector<float> &C, int N){

  float* A_d;
  float* B_d;
  float* C_d;

  Timer t_allocategpu;
  cudaMalloc((void**)(&A_d),N*N*sizeof(float));
  cudaMalloc((void**)(&B_d),N*N*sizeof(float));
  cudaMalloc((void**)(&C_d),N*N*sizeof(float));
  cudaDeviceSynchronize();
  std::cout << " Memories allocated on GPU! " << t_allocategpu.elapsed() << " secs."<< '\n';


  //Copy data to the GPU. 

  Timer t_copygpu;
  cudaMemcpy(A_d,A.data(),N*N*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(B_d,B.data(),N*N*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(C_d,C.data(),N*N*sizeof(float),cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  std::cout << " Data copied to GPU! " << t_copygpu.elapsed() << " secs."<< '\n';

  // //Call a GPU function (launch a grid of threads) gotta tell how many blocks and how many threads for block.
  dim3 numThreadsPerBlock(32,32); //(x,y,z)
  dim3 numBlocks((N+numThreadsPerBlock.x-1)/numThreadsPerBlock.x,(N+numThreadsPerBlock.y-1)/numThreadsPerBlock.y);
  
  Timer t_kernelgpu;
 // mm_kernel<<<numBlocks, numThreadsPerBlock>>>(A_d,B_d,C_d,N);
  std::cout << "GPU computation time " << t_kernelgpu.elapsed() << " secs."<< '\n';
  cudaDeviceSynchronize(); // wait for GPU to finish!

  //Copy from the GPU
  cudaMemcpy(C.data(),C_d,N*N,cudaMemcpyDeviceToHost);

  Timer t_deallocategpu;
  // Deallocate memory on the GPU
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
  cudaDeviceSynchronize();
  std::cout << " Memories freed for GPU! "<< t_deallocategpu.elapsed() <<  '\n';



}

int main() {

  int N {100};

  std::vector<float> A(N*N);
  std::vector<float> B(N*N);
  std::vector<float> C(N*N);

  for(int y {0}; y < N;++y){
    for(int x{0};x< N;++x){
    A.data()[y*N+x]= 0.1*Random::get(1,100);
  }}
  for(int y {0}; y < N;++y){
    for(int x{0};x< N;++x){
    B.data()[y*N+x]= 0.1*Random::get(1,100);
  }}
  std::cout << "matrix size is "<< A.size() << '\n';


  Timer t_cpu;
  mm_cpu(A,B,C,N*N);
  std::cout << " Cpu elapsed time " << t_cpu.elapsed() << '\n'; 

// Sample call: Random::get(1L, 6L);             // returns long
  Timer t_totalgpu;
  mm_gpu(A,B,C,N);
  cudaDeviceSynchronize();
  std::cout << " GPU elapsed time " << t_totalgpu.elapsed() << '\n'; 



  return 0;
}

#include <iostream>
#include "helper.h"
#include <vector>
#include <cstdlib>

void vecAddCpu(std::vector<float> &x, std::vector<float> &y,std::vector<float> &z, std::size_t N){
  for(std::size_t i{0};i<N;++i)
    z[i]=x[i]+y[i];
}

__global__ void vecAddKernel(float *x,float *y,float *z,std::size_t N){

}

void vecAddGpu(std::vector<float> &x, std::vector<float> &y,std::vector<float> &z, std::size_t N){
  //Allocate GPU memory so gotta write cudaError_t cudaMalloc(void **devPtr, size_t size)
  
  float * x_d;
  float * y_d;
  float * z_d;
  std::cout<< "before allocation " << x_d << '\n';

  //[IMPORTANT] So when you start float * x_d its just an adress in the CPU side but what you want is the x_d to hold is the adress on the GPU side. In order to change value of x_d to the GPU adress you need to feed in to cudaMalloc by its pointer so the parameters are a pointer to pointer. You do this because you want the exact adress (that is allocated) in the GPU because the next step is to cudaMemcopy which will copy the values on the CPU side (the data) to the adress on GPU which is x_d.
  
  cudaMalloc((void**)(&x_d),N*sizeof(float));
  cudaMalloc((void**)(&y_d),N*sizeof(float));
  cudaMalloc((void**)(&z_d),N*sizeof(float));

  std::cout<< "after allocation " << x_d << '\n';

  //Copy data to the GPU. Here x.data() is important because only x will have the adress of the vector object itself on the stack. But the x.data() will be the adress of the actual float numbers on the heap (probably the x[0] so that with the size info of each element it can get all the adresses.)

  cudaMemcpy(x_d,x.data(),N*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(y_d,y.data(),N*sizeof(float),cudaMemcpyHostToDevice);

  //Call a GPU function (launch a grid of threads) gotta tell how many blocks and how many threads for block.
  const unsigned int numThreadsPerBlock {512}; // again multiple of 32.
  const unsigned int numBlocks {N/512}; // since I have 512 threads per block and also N threads the N/512 will be number of blocks.
  vecAddKernel<<< numBlocks, numThreadsPerBlock >>>(x_d,y_d,z_d,N); // provide the configuration inside <<< >>>>. Now each thread will execute this function! (so no for loop or sth)

  //Copy from the GPU
  cudaMemcpy(z.data(),z_d,N*sizeof(float),cudaMemcpyDeviceToHost);

  // Deallocate memory on the GPU
  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(x_d);



}

int main(int argc, char **argv){
  Timer t;
  std::size_t N {(argc>1)?(std::stoul(argv[1])):(1<<25)}; // get default N=32 million or whatever you choose
  std::cout << N << '\n';
  std::vector<float> x(N); // allocate N sized vectors in cpp fashion the C fashion is (float*) malloc (N*sizeof(float))
  std::vector<float> y(N);
  std::vector<float> z(N);
  for(std::size_t i{0}; i < N;++i){ // populate the x and y vectors with random floats
    x[i]=rand();
    y[i]=rand();
  }
  //vector addition on CPU
  vecAddCpu(x,y,z,N);
  vecAddGpu(x,y,z,N);




  std::cout << "Time elapsed: " << t.elapsed() << " seconds for CPU computation\n";
  //vector addtion on GPU




  std::cout << "Time elapsed: " << t.elapsed() << " seconds for GPU computation\n";
  return 0;
}

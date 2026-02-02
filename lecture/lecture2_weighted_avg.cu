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

void rgb2gray_cpu(unsigned char* red,unsigned char* green, unsigned char* blue, unsigned char* gray, int width, int height){
  for(unsigned int i=0;i < width*height;++i)
   gray[i]= red[i]*3/10+green[i]*6/10+blue[i]*1/10;

}

__global__ void rgb2gray_kernel(unsigned char* red,unsigned char* green, unsigned char* blue, unsigned char* gray, int width, int height){
  unsigned int row= blockIdx.y*blockDim.y+threadIdx.y;
  unsigned int column= blockIdx.x*blockDim.x+threadIdx.x;
  unsigned int i = row*width+ column;
  if(row < height&& column < width)
   gray[i]= red[i]*3/10+green[i]*6/10+blue[i]*1/10;
}

void rgb2gray_gpu(std::vector<unsigned char> &red, std::vector<unsigned char> &green,std::vector<unsigned char> &blue, std::vector<unsigned char> &gray,unsigned int width, unsigned int height){

  unsigned char * red_d;
  unsigned char * green_d;
  unsigned char * blue_d;
  unsigned char * gray_d;
  unsigned int N{width*height};

  Timer t_allocategpu;
  cudaMalloc((void**)(&red_d),N*sizeof(unsigned char));
  cudaMalloc((void**)(&green_d),N*sizeof(unsigned char));
  cudaMalloc((void**)(&blue_d),N*sizeof(unsigned char));
  cudaMalloc((void**)(&gray_d),N*sizeof(unsigned int));
  cudaDeviceSynchronize();
  std::cout << " Memories allocated on GPU! " << t_allocategpu.elapsed() << " secs."<< '\n';


  //Copy data to the GPU. 

  Timer t_copygpu;
  cudaMemcpy(red_d,red.data(),N*sizeof(unsigned char),cudaMemcpyHostToDevice);
  cudaMemcpy(green_d,green.data(),N*sizeof(unsigned char),cudaMemcpyHostToDevice);
  cudaMemcpy(blue_d,blue.data(),N*sizeof(unsigned char),cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  std::cout << " Data copied to GPU! " << t_copygpu.elapsed() << " secs."<< '\n';

  // //Call a GPU function (launch a grid of threads) gotta tell how many blocks and how many threads for block.
  dim3 numThreadsPerBlock(32,32); //(x,y,z)
  dim3 numBlocks((width+numThreadsPerBlock.x-1)/numThreadsPerBlock.x,(height+numThreadsPerBlock.y-1)/numThreadsPerBlock.y);
  
  Timer t_kernelgpu;
  rgb2gray_kernel<<<numBlocks, numThreadsPerBlock>>>(red_d,green_d,blue_d,gray_d, width, height);
  std::cout << "GPU computation time " << t_kernelgpu.elapsed() << " secs."<< '\n';
  cudaDeviceSynchronize(); // wait for GPU to finish!

  //Copy from the GPU
  cudaMemcpy(gray.data(),gray_d,N*sizeof(unsigned char),cudaMemcpyDeviceToHost);

  Timer t_deallocategpu;
  // Deallocate memory on the GPU
  cudaFree(red_d);
  cudaFree(blue_d);
  cudaFree(green_d);
  cudaFree(gray_d);
  cudaDeviceSynchronize();
  std::cout << " Memories freed for GPU! "<< t_deallocategpu.elapsed() <<  '\n';



}

int main() {

  int width;
  int height;
  int channels;
  unsigned char *data =
      stbi_load("background.jpg", &width, &height, &channels, 0);

  // ... process data if not NULL ...
  // ... x = width, y = height, n = # 8-bit components per pixel ...
  // ... replace '0' with '1'..'4' to force that many components per pixel
  // ... but 'n' will always be the number that it would have been if you said 0
  if (data != nullptr)
    std::cout << "Loaded image: " << width << "x" << height << " with "
              << channels << " channels." << std::endl;

  int total_bytes = width * height * channels; // basically size of the array

  // get the red, green, blue vectors
  std::vector<unsigned char> red(total_bytes/channels);
  std::vector<unsigned char> green(total_bytes/channels);
  std::vector<unsigned char> blue(total_bytes/channels);
  std::vector<unsigned char> gray(total_bytes/channels);

  for (int i{0}; i < total_bytes; ++i) {
    if (i % 3 == 0)
      red[i/3] = data[i];
    else if(i%3==1)
      green[i/3] = data[i];
    else if(i%3==2)
      blue[i/3] = data[i];
  }

  Timer t_totalgpu;
  rgb2gray_gpu(red,green,blue,gray,width,height);
  cudaDeviceSynchronize();
  std::cout << " GPU elapsed time " << t_totalgpu.elapsed() << '\n'; 

  Timer t_cpu;
  rgb2gray_cpu(red.data(),green.data(),blue.data(),gray.data(),width,height);
  std::cout << " Cpu elapsed time " << t_cpu.elapsed() << '\n'; 

  stbi_write_png("output.png", width, height, 1, gray.data(), width * 1); // width
  stbi_image_free(data);

  return 0;
}

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

__device__ int blurSize {50}; // how big a pixel we choose for bluring

void blur_cpu(unsigned char* image, unsigned char* blurred, int width, int height,int blurSize) {
  for (int i = 0; i < width * height; ++i) {
    int outRow = i / width;
    int outCol = i % width;
    unsigned int average = 0; // Reset for every new pixel

    for (int inRow = outRow - blurSize; inRow < outRow + blurSize + 1; ++inRow) {
      for (int inCol = outCol - blurSize; inCol < outCol + blurSize + 1; ++inCol) {
        if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
          average += image[inRow * width + inCol];
        }
      }
    }
    // MOVED: Now happens only AFTER the neighbor loops finish
    blurred[outRow * width + outCol] = static_cast<unsigned char>(average / ((2 * blurSize + 1) * (2 * blurSize + 1)));
  }
}


__global__ void blur_kernel(unsigned char* image, unsigned char* blurred, int width, int height){

int outRow=blockIdx.y*blockDim.y+threadIdx.y;  // these are indices for the output obviously. then what we do is take the average of 9 pixels around that pixel
int outCol=blockIdx.x*blockDim.x+threadIdx.x;

if(outRow < height && outCol < width){
      unsigned int average {0};
  for(int inRow = outRow-blurSize;inRow < outRow+blurSize+1;++inRow){
    for(int inCol =outCol-blurSize;inCol< outCol+blurSize+1;++inCol){
      if(inRow >=0 && inRow < height && inCol >=0 && inCol < width){ // to check the edge cases where blurred is just at the edges then the ins for averages goes out of bounds
      average+= image[inRow*width+inCol]; }
      blurred[outRow*width+outCol]=static_cast<unsigned char>(average/((2*blurSize+1)*(2*blurSize+1)));}
      }}}




void blur_gpu(std::vector<unsigned char> &image, std::vector<unsigned char> &blurred,unsigned int width, unsigned int height){

  unsigned char * image_d;
  unsigned char * blurred_d;
  unsigned int N{width*height};

  Timer t_allocategpu;
  cudaMalloc((void**)(&image_d),N*sizeof(unsigned char));
  cudaMalloc((void**)(&blurred_d),N*sizeof(unsigned int));
  cudaDeviceSynchronize();
  std::cout << " Memories allocated on GPU! " << t_allocategpu.elapsed() << " secs."<< '\n';


  //Copy data to the GPU. 

  Timer t_copygpu;
  cudaMemcpy(image_d,image.data(),N*sizeof(unsigned char),cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  std::cout << " Data copied to GPU! " << t_copygpu.elapsed() << " secs."<< '\n';

  // //Call a GPU function (launch a grid of threads) gotta tell how many blocks and how many threads for block.
  dim3 numThreadsPerBlock(32,32); //(x,y,z)
  dim3 numBlocks((width+numThreadsPerBlock.x-1)/numThreadsPerBlock.x,(height+numThreadsPerBlock.y-1)/numThreadsPerBlock.y);
  
  Timer t_kernelgpu;
  blur_kernel<<<numBlocks, numThreadsPerBlock>>>(image_d,blurred_d, width, height);
  std::cout << "GPU computation time " << t_kernelgpu.elapsed() << " secs."<< '\n';
  cudaDeviceSynchronize(); // wait for GPU to finish!

  //Copy from the GPU
  cudaMemcpy(blurred.data(),blurred_d,N*sizeof(unsigned char),cudaMemcpyDeviceToHost);

  Timer t_deallocategpu;
  // Deallocate memory on the GPU
  cudaFree(image_d);
  cudaFree(blurred_d);
  cudaDeviceSynchronize();
  std::cout << " Memories freed for GPU! "<< t_deallocategpu.elapsed() <<  '\n';



}

int main() {

  int width;
  int height;
  int channels;
  unsigned char *data =
      stbi_load("doggo.jpg", &width, &height, &channels, 0);

  // ... process data if not NULL ...
  // ... x = width, y = height, n = # 8-bit components per pixel ...
  // ... replace '0' with '1'..'4' to force that many components per pixel
  // ... but 'n' will always be the number that it would have been if you said 0
  if (data != nullptr)
    std::cout << "Loaded image: " << width << "x" << height << " with "
              << channels << " channels." << std::endl;

  int total_bytes = width * height * channels; // basically size of the array

  // get the red, green, blue vectors
  std::vector<unsigned char> image(total_bytes/channels);
  std::vector<unsigned char> blurred(total_bytes/channels);

  for (int i{0}; i < total_bytes; ++i) {
    if (i % 3 == 0)
      image[i/3] = data[i];
    // else if(i%3==1)
    //   green[i/3] = data[i];
    // else if(i%3==2)
    //   blue[i/3] = data[i];
  }

  Timer t_cpu;
  blur_cpu(image.data(),blurred.data(),width,height,50);
  std::cout << " Cpu elapsed time " << t_cpu.elapsed() << '\n'; 

  Timer t_totalgpu;
  blur_gpu(image,blurred,width,height);
  cudaDeviceSynchronize();
  std::cout << " GPU elapsed time " << t_totalgpu.elapsed() << '\n'; 


  stbi_write_png("output_gray.png", width, height, 1, blurred.data(), width * 1); // width
  stbi_image_free(data);

  return 0;
}

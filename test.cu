#include <cstdio>

__global__ void hello() { printf("HELLO"); }

int main() {
  hello<<<1, 1>>>();
  cudaDeviceSynchronize();
}

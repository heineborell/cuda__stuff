#include <iostream>
int main() {
  int devID = 0;
  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, devID);
  int maxThreadsPerBlock = devProp.maxThreadsPerBlock;
  int maxThreadsPerMultiProcessor = devProp.maxThreadsPerMultiProcessor;
  std::cout << maxThreadsPerBlock << '\n';
  std::cout << maxThreadsPerMultiProcessor << '\n';

  return 0;
}

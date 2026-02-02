#include <vector>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>
// Basic usage (see HDR discussion below for HDR usage):
//    int x,y,n;
//    unsigned char *data = stbi_load(filename, &x, &y, &n, 0);
//    // ... process data if not NULL ...
//    // ... x = width, y = height, n = # 8-bit components per pixel ...
//    // ... replace '0' with '1'..'4' to force that many components per pixel
//    // ... but 'n' will always be the number that it would have been if you
//    said 0 stbi_image_free(data);
//
// Standard parameters:
//    int *x                 -- outputs image width in pixels
//    int *y                 -- outputs image height in pixels
//    int *channels_in_file  -- outputs # of image components in image file
//    int desired_channels   -- if non-zero, # of image components requested in
//    result
int main() {

  int width;
  int height;
  int channels;
  unsigned char *data =
      stbi_load("test_image.png", &width, &height, &channels, 0);

  // ... process data if not NULL ...
  // ... x = width, y = height, n = # 8-bit components per pixel ...
  // ... replace '0' with '1'..'4' to force that many components per pixel
  // ... but 'n' will always be the number that it would have been if you said 0
  if (data != nullptr)
    std::cout << "Loaded image: " << width << "x" << height << " with "
              << channels << " channels." << std::endl;

  int total_bytes = width * height * channels; // basically size of the array

  std::vector<char> output(total_bytes);
  for (int i{0}; i < total_bytes; ++i) {
    if (i % 3 == 1)
      output[i] = data[i];
    else
      output[i] = 0;
  }
  stbi_write_png("output.png", width, height, channels, output.data(),
                 width * channels);
  stbi_image_free(data);

  return 0;
}

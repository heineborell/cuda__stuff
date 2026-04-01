#include <iomanip> // for std::setprecision()
#include <iostream>

int main() {
  float f{123456789.0f};             // f has 10 significant digits
  std::cout << std::setprecision(9); // to show 9 digits in f
  std::cout << f << '\n';

  return 0;
}

#include <iomanip> // for output manipulator std::setprecision()
#include <iostream>

int main() {
  float xf{3.33333333333333333333333333333333333333f};
  double xd{3.33333333333333333333333333333333333333};
  std::cout << std::setprecision(17); // show 17 digits of precision
  std::cout << xf << '\n';            // f suffix means float
  std::cout << xd << '\n';            // no suffix means double

  return 0;
}

#include <iomanip> // for std::setprecision()
#include <iostream>

int main() {
  std::cout << std::setprecision(17);

  double d1{1.0};
  std::cout << d1 << '\n';

  double d2{0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 +
            0.1}; // should equal 1.0
  std::cout << d2 << '\n';
  if (d1 == d2)
    std::cout << " d1 is equal to d2 " << '\n';
  else
    std::cout << " d1 is not equal to d2 " << '\n';

  return 0;
}

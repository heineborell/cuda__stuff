#include <iostream>
void change(int *p) { *p = 8; }
int main() {
  int x{7};
  int *px{&x};
  change(px);
  std::cout << x;

  return 0;
}

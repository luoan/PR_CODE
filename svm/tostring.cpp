#include "tostring.h"

#include <algorithm>
#include <string>

std::string int2string(int n)
{
  std::string tmp;
  int i;
  char c;

  i = n%10;
  c = i-0 + 48;
  tmp += c;
  n /= 10;

  while (n > 0) {
    i = n % 10;
    c = i-0 + 48;
    tmp += c;
    n /= 10;
  }
  std::reverse(tmp.begin(), tmp.end());
  return tmp;
}

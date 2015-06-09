#include <iostream>
#include <vector>
using namespace std;

class A {
public:
  vector<int> v;
  A() {
    v.push_back(1);
    v.push_back(2);
    v.push_back(3);
  }
};
int main(int, char**) {
  A a1;
  A a2 = a1;
  a1.v[1] = 10;

  cout << "a1.v:" << a1.v[0] << ", " << a1.v[1] << ", " << a1.v[2] << endl;
  cout << "a2.v:" << a2.v[0] << ", " << a2.v[1] << ", " << a2.v[2] << endl;

  
}

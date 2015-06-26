#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "mtlib.h"

using namespace std;
using namespace cv;
using namespace mtlib;

class A {
public:
vector< vector<Point> > contours;
  A() {
    vector<Point> tmp;
    tmp.push_back(Point(1, 1));
    tmp.push_back(Point(1, 2));
    tmp.push_back(Point(1, 3));
    contours.push_back(tmp);
  }
};

double running_average(vector<int> data) {
  double average = 0;
  for (int i = 0; i < data.size(); i++) {
    average = data[i]/((double)i+1) + i*average/((double)(i+1));
  }
  return average;
}
double normal_average(vector<int> data) {
  int sum = 0;
  for (int i = 0; i < data.size(); i++) sum += data[i];
  return ((double)sum)/data.size();
}
typedef struct {
  unsigned long real_time;
  clock_t clock_cycles;
} prog_time_t;
unsigned long time_milliseconds() {
  return chrono::duration_cast<chrono::milliseconds>
    (chrono::system_clock::now().time_since_epoch()).count();
}
prog_time_t current_clock() {
  prog_time_t t;
  t.real_time = time_milliseconds();
  t.clock_cycles = clock();
  return t;
}
ostream& operator << (ostream &o, const prog_time_t t) {
  o << "[real: " << t.real_time << ", clock: " 
    << (((float)t.clock_cycles)/CLOCKS_PER_SEC) << "]";
  return o;
}
prog_time_t operator - (prog_time_t t1, prog_time_t t2) {
  prog_time_t o;
  o.real_time = t1.real_time - t2.real_time;
  o.clock_cycles = t1.clock_cycles - t2.clock_cycles;
  return o;
}
int main(int, char**) {
  /////////////////////////////////////////////////////////
  // A a1 = A();                                         //
  // A a2 = a1;                                          //
  // a2.contours[0][0] = Point(10, 10);                  //
  // for (int i = 0; i < a1.contours.size(); i++) {      //
  //   for (int j = 0; j < a1.contours[i].size(); j++) { //
  //     cout << a1.contours[i][j] << ", ";              //
  //   }                                                 //
  //   cout << endl;                                     //
  // }                                                   //
  // for (int i = 0; i < a2.contours.size(); i++) {      //
  //   for (int j = 0; j < a2.contours[i].size(); j++) { //
  //     cout << a2.contours[i][j] << ", ";              //
  //   }                                                 //
  //   cout << endl;                                     //
  // }                                                   //
  /////////////////////////////////////////////////////////
  VideoWriter output ("test_video_file.mp4", 
                      -1,
                      30,
                      Size(640, 480));

  Mat tmp(Size(640, 480), CV_8UC3, Scalar(255, 0, 255));
  prog_time_t t;
  for (int i = 0; i < 100; i++) {

    t = current_clock();
    output.write(tmp);
    t = current_clock() - t;
    cout << i << " " << t << endl;
  }
  
}

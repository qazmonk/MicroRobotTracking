#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include "mtlib.h"
#include <algorithm>
#include <time.h>
#include "firefly.h"
#include <sstream>
#include <sys/stat.h>

using namespace std;
using namespace cv;
using namespace mtlib;

firefly_t * f;
firefly_camera_t * camera;
int a;

int capture_from_camera(Mat * dst) {
  return opencv_firefly_capture(camera, dst);
}

void save_frame(Mat frame) {
  const char* prefix = "image-";
  const char* suffix = ".png";
  int count = 0;
  char buffer [100];
  sprintf(buffer, "%s%.4d%s", prefix, count, suffix);
  while (file_exists(buffer)) {
    count++;
    sprintf(buffer, "%s%.4d%s", prefix, count, suffix);
  }
  cout << "Writing frame " << buffer << endl;
  imwrite(buffer, frame);
}
int main(int argc, char* argv[]) {
  f = firefly_new();
  firefly_setup_camera(f, &camera);
  firefly_start_transmission(camera);
  Mat frame, dst;
  namedWindow("Camera", CV_WINDOW_AUTOSIZE);
  while (true) {
    int rc = capture_from_camera(&frame);
    imshow("Camera", frame);
    if (waitKey(1000/60) == ' ') {
      save_frame(frame);
    }
  }
  firefly_stop_transmission(camera);
  firefly_cleanup_camera(camera);
  firefly_free(f);
  return 0;

}

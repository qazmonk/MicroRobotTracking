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

using namespace std;
using namespace cv;
using namespace mtlib;

firefly_t * f;
firefly_camera_t * camera;

int capture_from_camera(Mat * dst) {
  return opencv_firefly_capture(camera, dst);
}

int main(int argc, char* argv[]) {
  Model::init();
  f = firefly_new();
  firefly_setup_camera(f, &camera);
  firefly_start_transmission(camera);
  namedWindow("Output", CV_WINDOW_AUTOSIZE);
  namedWindow("Input", CV_WINDOW_AUTOSIZE);
  Mat frame, dst;
  vector< vector<Point> > contours;
  while (waitKey(1000/60) == -1) {
    int rc = capture_from_camera(&frame);
    filterAndFindContours(frame.clone(), &contours);
    dst = Mat::zeros(frame.size(), CV_8UC3);
    drawContoursFast(dst, &contours, 500, 25000);
    imshow("Output", dst);
    imshow("Input", frame);
  }
  firefly_stop_transmission(camera);
  firefly_cleanup_camera(camera);
  firefly_free(f);
  return 0;
}

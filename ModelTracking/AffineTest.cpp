#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/nonfree/nonfree.hpp"
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

vector<Mat> video;
int fps, ex;
Size S;
Mat dmd_img;
firefly_t * f;
firefly_camera_t * camera;

int capture_no_camera(Mat * frame) {
  *frame = dmd_img;
  return 0;
}
int capture_camera(Mat * dst) {
  return opencv_firefly_capture(camera, dst);
}
int main(int argc, char* argv[]) {
  //captureVideo(argv[1], &video, &fps, &S, &ex);
  bool using_camera = false;
  int dmd_w = 608, dmd_h = 662, dmd_x = 39, dmd_y = 1200;
  Mat src_img;  
  
  for (int i = 0; i < argc; i++) {
    if (strncmp(argv[i], "--camera", 10) == 0) {
      using_camera = true;
    } else if(strncmp(argv[i], "--file", 10) == 0) {
      src_img = imread(argv[i+1], CV_LOAD_IMAGE_COLOR);
      dmd_img = imread(argv[i+2], CV_LOAD_IMAGE_COLOR);      
      i += 2;
    } else if (strncmp(argv[i], "--affine-args", 15) == 0) {
      dmd_w = stoi(argv[i+1]);
      dmd_h = stoi(argv[i+2]);
      dmd_x = stoi(argv[i+3]);
      dmd_y = stoi(argv[i+4]);
      i += 4;
    }
  }
  namedWindow("DMD", CV_WINDOW_NORMAL);
  cvMoveWindow("DMD", dmd_x, dmd_y);
  cvResizeWindow("DMD", dmd_w, dmd_h);
  cout << "Resized DMD window to: " << Size(dmd_w, dmd_h) << endl;
  Mat white(Size(dmd_w, dmd_h), CV_8UC3, Scalar(255, 255, 255));
  imshow("DMD", white);


  vector<Point2f> pts;
  if (using_camera) {
    f = firefly_new();
    firefly_frame_t frame;
    firefly_setup_camera(f, &camera);
    firefly_start_transmission(camera);
    firefly_capture_frame(camera);
    usleep(10000);
    frame = firefly_capture_frame(camera);
    src_img = frame.img;
    pts = getAffineTransformPoints(src_img, *capture_camera, "DMD", dmd_w, dmd_h);
  } else {
    pts = getAffineTransformPoints(src_img, *capture_no_camera, "DMD", dmd_w, dmd_h);
  }



  Point2f src_pts[] = {pts[0], pts[1], pts[2]};
  Point2f dst_pts[] = {pts[3], pts[4], pts[5]};
  Mat warp_mat = getAffineTransform(dst_pts, src_pts);

  Mat warped(Size(dmd_w, dmd_h), CV_8UC3);
  Mat src_filtered, src_gray, src_inv;
  cvtColor(src_img, src_gray, CV_BGR2GRAY);
  adaptiveThreshold(src_gray, src_filtered, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 27, 5);
  bitwise_not(src_filtered, src_inv);
  namedWindow("Result", CV_WINDOW_AUTOSIZE);
  warpAffine(expandForDMD(src_filtered, dmd_w, dmd_h), warped, warp_mat, warped.size(),
             INTER_LINEAR, BORDER_CONSTANT, Scalar(255, 255, 255));
  imshow("Result", warped);
  imshow("DMD", warped);
  usleep(1000000);
  Mat result;
  if (using_camera) {
    while (waitKey(1000/60) == -1) {
      capture_camera(&result);
      imshow("Result", result);
    }
  }
  waitKey(0);
  if (using_camera) {
    firefly_stop_transmission(camera);
    firefly_cleanup_camera(camera);
    firefly_free(f);
  }
}

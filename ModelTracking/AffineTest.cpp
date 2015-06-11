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

Mat capture_no_camera() {
  return dmd_img;
}
Mat capture_camera() {
  firefly_frame_t frame = firefly_capture_frame(camera);
  if (frame.frames_behind > 0) {
      firefly_flush_camera(camera);
      firefly_start_transmission(camera);
      frame = firefly_capture_frame(camera);
  }
  return frame.img;
}
int main(int argc, char* argv[]) {
  //captureVideo(argv[1], &video, &fps, &S, &ex);
  bool using_camera = false;
  for (int i = 0; i < argc; i++) {
    if (strncmp(argv[i], "--camera", 2) == 0) {
      using_camera = true;
    }
  }
  Mat src_img;
  vector<Point> pts;
  if (using_camera) {
    f = firefly_new();
    firefly_frame_t frame;
    firefly_setup_camera(f, &camera);
    firefly_start_transmission(camera);
    frame = firefly_capture_frame(camera);
    src_img = frame.img;
    pts = getAffineTransformPoints(src_img, *capture_camera, 800, 500, 400, 400);
  } else {
    src_img = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    dmd_img = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    pts = getAffineTransformPoints(src_img, *capture_no_camera, 800, 500, 400, 400);
  }



  Point2f src_pts[] = {pts[0], pts[1], pts[2]};
  Point2f dst_pts[] = {pts[3], pts[4], pts[5]};
  Mat warp_mat = getAffineTransform(dst_pts, src_pts);

  Mat warped;
  namedWindow("output", CV_WINDOW_AUTOSIZE);
  warpAffine(dmd_img, warped, warp_mat, warped.size());
  imshow("output", warped);
  waitKey(0);
  if (using_camera) {
    firefly_stop_transmission(camera);
    firefly_cleanup_camera(camera);
    firefly_free(f);
  }
}

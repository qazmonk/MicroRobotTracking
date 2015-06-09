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

using namespace std;
using namespace cv;
using namespace mtlib;

vector<Mat> video;
int fps, ex;
Size S;
Mat dmd_img;
Mat capture() {
  return dmd_img;
}
int main(int argc, char* argv[]) {
  //captureVideo(argv[1], &video, &fps, &S, &ex);
  Mat src_img = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  dmd_img = imread(argv[2], CV_LOAD_IMAGE_COLOR);
  vector<Point> pts = getAffineTransformPoints(src_img, *capture, 800, 500, 400, 400);
  Point2f src_pts[] = {pts[0], pts[1], pts[2]};
  Point2f dst_pts[] = {pts[3], pts[4], pts[5]};
  Mat warp_mat = getAffineTransform(dst_pts, src_pts);

  Mat warped;
  namedWindow("output", CV_WINDOW_AUTOSIZE);
  warpAffine(dmd_img, warped, warp_mat, warped.size());
  imshow("output", warped);
  waitKey(0);

}

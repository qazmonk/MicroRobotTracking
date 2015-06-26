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

int main(int argc, char* argv[]) {
  Mat src = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  Mat dst = imread(argv[2], CV_LOAD_IMAGE_COLOR);
  
  vector<Point> src_pts, dst_pts;
  namedWindow("Source", CV_WINDOW_AUTOSIZE);
  namedWindow("Destination", CV_WINDOW_AUTOSIZE);
  imshow("Source", src);
  imshow("Destination", dst);
  getNPoints(4, "Source", &src_pts, src);
  getNPoints(4, "Destination", &dst_pts, dst);
  vector<Point2f> src_pts2f, dst_pts2f;
  for (int i = 0; i < 4; i++) {
    src_pts2f.push_back(Point2f(src_pts[i].x, src_pts[i].y));
    dst_pts2f.push_back(Point2f(dst_pts[i].x, dst_pts[i].y));
  }
  Mat warp_mat = getPerspectiveTransform(dst_pts2f, src_pts2f);

  Mat warped(src.size(), src.type());
  cout << warp_mat << endl;
  warpPerspective(dst, warped, warp_mat, warped.size());
  namedWindow("Result", CV_WINDOW_AUTOSIZE);
  imshow("Result", warped);
  waitKey(0);
}

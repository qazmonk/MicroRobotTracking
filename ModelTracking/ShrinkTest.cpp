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

int main(int argc, char* argv[]) {

  captureVideo(argv[1], &video, &fps, &S, &ex);

  
  namedWindow("Shrink", CV_WINDOW_AUTOSIZE);
  namedWindow("Contours", CV_WINDOW_AUTOSIZE);
  namedWindow("Original", CV_WINDOW_AUTOSIZE);
  imshow("Original", video[0]);
  Mat out = video[0];
  
  while (1) {
  
    imshow("Shrink", out);

    vector< vector<Point> > contours;
    vector< Vec4i > hierarchy;
    
    filterAndFindContoursElizabeth(out, &contours, &hierarchy);
    Mat dst = Mat::zeros(out.size(), CV_8UC3);

    int min = -1, max = 0;
    for (int i = 0; i < contours.size(); i++) {
      double consize = contourArea(contours.at(i));
      if (min == -1 || consize < min) min = consize;
      if (max < consize) max = consize;
    }
    drawContoursAndFilter(dst, &contours, &hierarchy, (int)((max-min)*0.5 + min), max);

    imshow("Contours", dst);
    out = fourToOne(out);
    waitKey(0);
  }
}

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <algorithm>
#include <time.h>
#include <sstream>
#include <sys/stat.h>
#include "mtlib.h"
#include <ostream>

using namespace std;
using namespace cv;
using namespace mtlib;

int fps, ex, pos = 0;
Size S;
vector<Mat> video;  
int main(int argc, char* argv[]) {
  Mat frame;
  for (int i = 1; i < argc; i++) {
    if (strncmp(argv[i], "--video", 10) == 0) {
      captureVideo(argv[i+1], &video, &fps, &S, &ex);
      frame = video[stoi(argv[i+2])];
      i += 2;
    } else if (strncmp(argv[i], "--picture", 15) == 0) {
      frame = imread(argv[i+1], CV_LOAD_IMAGE_COLOR);
      i++;
    }
  }
  /*save_frame_safe(frame, "original", ".png");
  vector<Mat> rgb;
  split(frame, rgb);
  rgb[0].setTo(0);
  merge(rgb, frame);
  save_frame_safe(frame, "no_blue", ".png");
  Mat gray;
  cvtColor(frame, gray, CV_BGR2GRAY);
  save_frame_safe(gray, "grayscale", ".png");
  Mat thresh;
  adaptiveThreshold(gray, thresh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 27, 5);
  save_frame_safe(thresh, "adaptive_thresh", ".png");
  Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
  Mat eroded;
  erode(thresh, eroded, element);
  threshold(eroded, eroded, 40, 255, THRESH_BINARY);
  save_frame_safe(eroded, "eroded", ".png");

  vector< vector<Point> > contours;
  filterAndFindContours(frame, &contours);
  Mat contour_img = Mat::zeros(frame.size(), CV_8UC3);
  drawContoursFast(contour_img, &contours, 1000, 25000);
  save_frame_safe(contour_img, "contours", ".png");*/


}

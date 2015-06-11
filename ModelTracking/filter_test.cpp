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

using namespace std;
using namespace cv;
using namespace mtlib;

vector<Mat> video;
vector<Mat> out;
int fps, ex, pos = 0;
Size S;
string window = "Output";
int skip = 1;
vector<Model> models;

void scrub(int, void*);

int main(int argc, char* argv[]) {
  cout << "reading file..." << flush;
  captureVideo(argv[1], &video, &fps, &S, &ex);
  cout << "done" << endl;
  int minArea = -1, maxArea = -1;
  int startFrame = 0;
  for (int i = 0; i < argc; i++) {
    if (strncmp(argv[i], "-c", 2) == 0) {
      mtlib::setDefaultChannel(stoi(argv[i+1]));
      i++;
    } else if (strncmp(argv[i], "--bounds", 10) == 0) {
      minArea = stoi(argv[i+1]);
      maxArea = stoi(argv[i+2]);
      i+=2;
    } else if (strncmp(argv[i], "-s", 5) == 0) {
      startFrame = stoi(argv[i+1]);
      i++;
    }
  }
  if (minArea < 0 || maxArea < 0 || minArea > maxArea) {
    Point minMax = getMinAndMaxAreas(video[startFrame]);
    minArea = minMax.x;
    maxArea = minMax.y;
  }
  namedWindow("test", CV_WINDOW_AUTOSIZE);
  for (int i = startFrame; i < video.size(); i++) {
    cout << "processing frame " << i << " of " << (video.size()-1) << endl;;
    Mat frame = video[i];
    Mat filtered;
    mtlib::filter(filtered, frame);
    Mat filtered_color;
    cvtColor(filtered, filtered_color, CV_GRAY2RGB);
    vector< vector<Point> > contours;
    filterAndFindContours(frame, &contours);
    Mat contour_img = Mat::zeros(frame.size(), frame.type());
    drawContoursFast(contour_img, &contours, minArea, maxArea);
    Mat comb_top, out_img;
    combineHorizontal(comb_top, frame, contour_img);
    combineVertical(out_img, comb_top, filtered_color);
    out.push_back(out_img);
  }
  namedWindow("Output", CV_WINDOW_AUTOSIZE);
  createTrackbar("Scrubbing", "Output", &pos, out.size()-1, scrub);
  scrub(0, 0);
  waitKey(0);
}

void scrub (int , void* ) {
  imshow("Output", out[pos]);
}

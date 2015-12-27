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
char buff[50];

const char * make_filename(const char * prefix, const char * name) {
  sprintf(buff, "%s-%s", prefix, name);
  return buff;
}
int main(int argc, char* argv[]) {
  Mat frame;
  const char * output_prefix = argv[1];
  for (int i = 1; i < argc; i++) {
    if (strncmp(argv[i], "--video", 10) == 0) {
      int frame_num = stoi(argv[i+2]);
      captureVideo(argv[i+1], &video, &fps, &S, &ex, frame_num+1);
      frame = video[frame_num];
      i += 2;
    } else if (strncmp(argv[i], "--picture", 15) == 0) {
      frame = imread(argv[i+1], CV_LOAD_IMAGE_COLOR);
      i++;
    }
  }
  vector<mtlib::Model> models;
  int minArea = 500, maxArea = 25000;
  mtlib::generateModels(frame, &models, minArea, maxArea);
  vector<int> selected =  mtlib::selectObjects(frame, &models);
  vector<mtlib::Model> selectedModels;
  for (int i = 0; i < selected.size(); i++) {
    selectedModels.push_back(models[selected[i]]);
  }
  models = selectedModels;
  mtlib::set_output_figure_name(make_filename(output_prefix, "roi-identified"));
  mtlib::updateModels(frame, &models, minArea, maxArea);
  //save_frame_safe(frame, make_filename(output_prefix, "original"), ".png");
  /*Mat gray, blurred, edges;
  cvtColor(frame, gray, CV_BGR2GRAY);
  
  blur(gray, blurred, Size(3, 3));
  int low_threshold = 5;
  Canny(blurred, edges, low_threshold, low_threshold*3, 3);

  Mat edges_color, comb;
  cvtColor(edges, edges_color, CV_GRAY2BGR);
  combineHorizontal(comb, frame, edges_color);
  save_frame_safe(comb, make_filename(output_prefix, "canny"), ".png");
  /*vector<Mat> rgb;
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

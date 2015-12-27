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
  int minArea = -1, maxArea = -1;
  int startFrame = 0, endFrame = 0;
  bool debug = false;

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
    } else if (strncmp(argv[i], "-e", 5) == 0) {
      endFrame = stoi(argv[i+1]);
      i++;
    } else if (strncmp(argv[i], "--debug", 10) == 0) {
      debug = true;
      i++;
    }
  }
  if (endFrame <= startFrame) {
    cout << "reading file..." << flush;
    captureVideo(argv[1], &video, &fps, &S, &ex);
    cout << "done" << endl;
  } else {
    cout << "reading file..." << flush;
    captureVideo(argv[1], &video, &fps, &S, &ex, endFrame);
    cout << "done" << endl;
  }
  if (minArea < 0 || maxArea < 0 || minArea > maxArea) {
    Point minMax = getMinAndMaxAreas(video[startFrame]);
    minArea = minMax.x;
    maxArea = minMax.y;
  }
  for (int i = startFrame; i < video.size(); i++) {
    cout << "processing frame " << i << " of " << (video.size()-1) << endl;;
    Mat frame = video[i];
    Mat filtered;
    vector<Mat> filter_frames = filter_debug(filtered, frame);
    int dim = ceil(sqrt(filter_frames.size()));
    vector<Mat> horizontal;
    for (int i = 0; i < dim; i++) {
      vector<Mat> tmp;
      for (int j = 0; j < dim; j++) {
        if (i*dim + j < filter_frames.size()) {
          Mat cur = filter_frames[i*dim+j];
          if (j > 0) {
            Mat comb;
            combineHorizontal(comb, tmp.back(), cur);
            tmp.push_back(comb.clone());
          } else {
            tmp.push_back(cur);
          }
        }
      }
      if (tmp.size() > 0) {
        horizontal.push_back(tmp.back());
      }
    }
    Mat tmp = horizontal[0];
    for (int i = 1; i < horizontal.size(); i++) {
      Mat tmp2;
      combineVertical(tmp2, tmp, horizontal[i]);
      tmp = tmp2.clone();
    }
    Mat tmp_scaled;
    resize(tmp, tmp_scaled, Size(), 0.75, 0.75, CV_INTER_AREA);
    out.push_back(tmp_scaled);
  }
  if (!debug) {
    namedWindow("Output", CV_WINDOW_AUTOSIZE);
    createTrackbar("Scrubbing", "Output", &pos, out.size()-1, scrub);
    scrub(0, 0);
    waitKey(0);
  }
}

void scrub (int , void* ) {
  imshow("Output", out[pos]);
}

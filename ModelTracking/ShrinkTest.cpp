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

vector<Mat> video, out;
int fps, ex, pos;
Size S;

void scrub(int, void*);

int main(int argc, char* argv[]) {

  captureVideo(argv[1], &video, &fps, &S, &ex);

  bool write = false;
  char* output_folder;
  for (int i = 2; i < argc; i++) {
    if (strncmp(argv[i], "-c", 2) == 0) {
      mtlib::setDefaultChannel(stoi(argv[i+1]));
      i++;
    } else if (strncmp(argv[i], "-w", 3) == 0) {
      write = true;
      output_folder = argv[i+1];
      i++;
    }
  }

  
  namedWindow("Shrink", CV_WINDOW_AUTOSIZE);
  namedWindow("Contours", CV_WINDOW_AUTOSIZE);
  namedWindow("Original", CV_WINDOW_AUTOSIZE);
  imshow("Original", video[0]);
  
  for (int i = 0; i < video.size(); i++) {
    Mat shrunk = fourToOne(fourToOne(video[i]));
    
    vector<Vec2f> lines;
    filterAndFindLines(shrunk, &lines);
    

    Mat dst;
    cvtColor(filter(shrunk), dst, CV_GRAY2RGB);
    //drawLines(dst, &lines);
    
    vector< vector<Point> > contours;
    vector< Vec4i > hierarchy;
    filterAndFindContoursElizabeth(shrunk, &contours, &hierarchy);
    int min = -1, max = 0;
    for (int i = 0; i < contours.size(); i++) {
      double consize = contourArea(contours.at(i));
      if (min == -1 || consize < min) min = consize;
      if (max < consize) max = consize;
    }
    drawContoursAndFilter(dst, &contours, &hierarchy, (int)((max-min)*0.5 + min), max);
    out.push_back(dst);
  }
  if (write) writeVideo(output_folder, out);
  createTrackbar("Scrubbing", "Contours", &pos, video.size()-1, scrub);
  scrub(0, 0);
  waitKey(0);
}

void scrub (int , void* ) {
  imshow("Contours", out[pos]);
}

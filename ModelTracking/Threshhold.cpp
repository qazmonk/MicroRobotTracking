#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
//#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include "mtlib.h"
#include <algorithm>
#include <time.h>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <climits>
#include "ttmath/ttmath.h"

using namespace std;
using namespace cv;
using namespace mtlib;

int fps, ex, pos;
Size S;

int thresh, frame;
const char* window = "Video";
vector<Mat> video;

void scrub(int, void*);


int main(int argc, char* argv[]) {

  captureVideo(argv[1], &video, &fps, &S, &ex);

  namedWindow(window, CV_WINDOW_AUTOSIZE);
  createTrackbar("Scrub", window, &pos, video.size()-1, scrub);
  createTrackbar("Thresh", window, &thresh, 255, scrub);
  waitKey(0);
}


void scrub(int, void*) {
  Mat dst, gray;

  cvtColor(video[pos], gray, CV_RGB2GRAY);
  threshold(gray, dst, thresh, 255, THRESH_TOZERO);
  
  imshow(window, dst);
}

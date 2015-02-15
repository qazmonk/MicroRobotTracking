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

Mat capture() {
  return video[10];
}
int main(int argc, char* argv[]) {
  captureVideo(argv[1], &video, &fps, &S, &ex);
  getAffineTransformPoints(video[0], *capture, 640, 480, 400, 400);
}

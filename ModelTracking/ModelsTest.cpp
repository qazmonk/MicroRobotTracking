#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include "mtlib.h"
#include <algorithm>


using namespace std;
using namespace cv;
using namespace mtlib;

vector<Mat> video;
vector<Mat> out;
int fps, ex, pos = 0;
Size S;
string window = "Output";

vector<Model> models;

void scrub(int, void*);

int main(int argc, char* argv[]) {


  captureVideo(argv[1], &video, &fps, &S, &ex);
  

  Point minMax = getMinAndMaxAreas(video[0]);

  int minArea = minMax.x;
  int maxArea = minMax.y;
  out.reserve(video.size());


  mtlib::generateModels(video[0], &models, minArea, maxArea);
  mtlib::selectObjects(video[0], &models);
  namedWindow(window, CV_WINDOW_AUTOSIZE);

  for (int i = 0; i < video.size(); i++) {


    if (i > 0) {
      cout << "calculating frame " << i << endl;
      updateModels(video[i], &models, minArea, maxArea);

    }

    vector< vector<Point> > contours;
    vector< Vec4i > hierarchy;
    

    filterAndFindContours(video[i], &contours, &hierarchy);
    
    Mat dst = Mat::zeros(video[i].size(), CV_8UC1);


    drawContoursAndFilter(dst, &contours, &hierarchy, minArea, maxArea);

    for (int n = 0; n < models.size(); n++)
      models[n].drawModel(dst, i);


    out.push_back(dst);
    
  }

  writeFile("data.txt", models);
    
  createTrackbar("Scrubbing", window, &pos, video.size()-1, scrub);
  scrub(0, 0);

  waitKey(0);
}

void scrub (int , void* ) {
  imshow(window, out[pos]);
}

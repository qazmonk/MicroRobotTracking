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
vector<Mat> out;
int fps, ex, pos = 0;
Size S;
string window = "Output";
int skip = 1;
vector<Model> models;

void scrub(int, void*);

int main(int argc, char* argv[]) {
  cout << "reading file..." << endl;
  captureVideo(argv[1], &video, &fps, &S, &ex);
  cout << "done" << endl;
  bool write = false, partialComp = true;
  int numFrames = video.size(), startFrame = 0;
  char* output_folder;
  int minArea = -1;
  int maxArea = -1;

  for (int i = 0; i < argc; i++) {
    if (strncmp(argv[i], "-c", 2) == 0) {
      mtlib::setDefaultChannel(stoi(argv[i+1]));
      i++;
    } else if (strncmp(argv[i], "-w", 3) == 0) {
      write = true;
      output_folder = argv[i+1];
      i++;
    } else if (strncmp(argv[i], "-n", 3) == 0) {
      partialComp = true;
      numFrames = startFrame+stoi(argv[i+1]);
      i++;
    } else if (strncmp(argv[i], "-s", 3) == 0) {
      startFrame = stoi(argv[i+1]);
      i++;
    } else if (strncmp(argv[i], "-e", 3) == 0) {
      numFrames = stoi(argv[i+1]);
      i++;
    } else if (strncmp(argv[i], "--bounds", 10) == 0) {
      minArea = stoi(argv[i+1]);
      maxArea = stoi(argv[i+2]);
      i+=2;
    }
  }

  /*int tmpa[] = {1, 2 ,3};
  vector<double> tmpv(tmpa,tmpa+sizeof(tmpa)/sizeof(int));
  namedWindow("tmp", CV_WINDOW_AUTOSIZE);
  showHist("tmp", tmpv);*/

  if (minArea < 0 || maxArea < 0 || minArea > maxArea) {
    Point minMax = getMinAndMaxAreas(video[0]);
    minArea = minMax.x;
    maxArea = minMax.y;
  }
  out.reserve(video.size());
 
  cout << "generating models...";
  mtlib::generateModels(video[0], &models, minArea, maxArea);
  cout << "done" << endl;
  vector<int> selected =  mtlib::selectObjects(video[0], &models);
  vector<mtlib::Model> selectedModels;
  for (int i = 0; i < selected.size(); i++) {
    selectedModels.push_back(models[selected[i]]);
  }
  models = selectedModels;
  namedWindow(window, CV_WINDOW_AUTOSIZE);


  clock_t t;
  for (int i = startFrame; i < numFrames; i += skip) {

    t = clock();
    
    if (i > 0) {
      cout << "calculating frame " << i << endl;
      updateModels(video[i], &models, minArea, maxArea);
    }

    vector< vector<Point> > contours;
    vector< Vec4i > hierarchy;
    

    filterAndFindContours(video[i], &contours, &hierarchy);
    
    Mat dst = Mat::zeros(video[i].size(), CV_8UC3);


    drawContoursAndFilter(dst, &contours, &hierarchy, minArea, maxArea);

    for (int n = 0; n < models.size(); n++) {
      int t = models[n].curTime();
      models[n].drawBoundingBox(dst, t, Scalar(255, 0, 0));
      models[n].drawModel(dst, t);
    }
    t = clock() - t;
    printf("It took %f seconds to calculate that frame\n", ((float)t)/CLOCKS_PER_SEC);
    out.push_back(dst);
    
  }
  
  if (write) { writeVideo(output_folder, out); }

  createTrackbar("Scrubbing", window, &pos, out.size()-1, scrub);
  
  cout << "size: " << out.size()-1 <<  " val: " << getTrackbarPos("Scrubbing", window) << endl;
  scrub(0, 0);

  waitKey(0);
}



void scrub (int , void* ) {
  imshow(window, out[pos]);
}

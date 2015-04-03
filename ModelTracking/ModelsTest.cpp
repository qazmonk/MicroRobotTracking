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
int skip = 2;
vector<Model> models;

void scrub(int, void*);

int main(int argc, char* argv[]) {

  bool write = false;
  char* output_folder;
  for (int i = 0; i < argc; i++) {
    if (strncmp(argv[i], "-c", 2) == 0) {
      mtlib::setDefaultChannel(stoi(argv[i+1]));
      i++;
    } else if (strncmp(argv[i], "-w", 3) == 0) {
      write = true;
      output_folder = argv[i+1];
      i++;
    }
  }
  captureVideo(argv[1], &video, &fps, &S, &ex);
  

  //Point minMax = getMinAndMaxAreas(video[0]);

  int minArea = 12000;//minMax.x;
  int maxArea = 20000;//minMax.y;
  out.reserve(video.size());

  mtlib::generateModels(video[0], &models, minArea, maxArea);
  //vector<int> selected =  mtlib::selectObjects(video[0], &models);
  vector<mtlib::Model> selectedModels;
  /*for (int i = 0; i < selected.size(); i++) {
    selectedModels.push_back(models[i]);
    }*/
  selectedModels.push_back(models[0]);
  models = selectedModels;
  //namedWindow(window, CV_WINDOW_AUTOSIZE);

  clock_t t;
  for (int i = 0; i < video.size(); i += skip) {

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
  
  if (write) writeVideo(output_folder, out);
  //writeFile("data.txt", models);
    
  //createTrackbar("Scrubbing", window, &pos, video.size()-1, scrub);
  //scrub(0, 0);

  //waitKey(0);
}



void scrub (int , void* ) {
  imshow(window, out[pos/skip]);
}

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
#include <sys/stat.h>

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
  Model::init();

  bool write = false, partialComp = true, write_hq = false;
  int endFrame = 0, startFrame = 0;
  char* output_folder, *video_folder;
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
      endFrame = startFrame+stoi(argv[i+1]);
      i++;
    } else if (strncmp(argv[i], "-s", 3) == 0) {
      startFrame = stoi(argv[i+1]);
      i++;
    } else if (strncmp(argv[i], "-e", 3) == 0) {
      endFrame = stoi(argv[i+1]);
      i++;
    } else if (strncmp(argv[i], "--bounds", 10) == 0) {
      minArea = stoi(argv[i+1]);
      maxArea = stoi(argv[i+2]);
      i+=2;
    } else if (strncmp(argv[i], "--hqvideo", 10) == 0) {
      video_folder = argv[i+1];
      write_hq = true;
      i++;
    }
  }
  cout << "reading file..." << endl;
  if (endFrame <= startFrame) {
    captureVideo(argv[1], &video, &fps, &S, &ex);
    endFrame = video.size();
  } else {
    captureVideo(argv[1], &video, &fps, &S, &ex, endFrame);
  }
  cout << "done" << endl;
  /*int tmpa[] = {1, 2 ,3};
  vector<double> tmpv(tmpa,tmpa+sizeof(tmpa)/sizeof(int));
  namedWindow("tmp", CV_WINDOW_AUTOSIZE);
  showHist("tmp", tmpv);*/

  if (minArea < 0 || maxArea < 0 || minArea > maxArea) {
    Point minMax = getMinAndMaxAreas(video[startFrame]);
    minArea = minMax.x;
    maxArea = minMax.y;
  }
  out.reserve(video.size());
 
  cout << "generating models..." << flush;
  mtlib::generateModels(video[startFrame], &models, minArea, maxArea);
  cout << "done" << endl;
  vector<int> selected =  mtlib::selectObjects(video[startFrame], &models);
  vector<mtlib::Model> selectedModels;
  for (int i = 0; i < selected.size(); i++) {
    selectedModels.push_back(models[selected[i]]);
  }
  models = selectedModels;
  namedWindow(window, CV_WINDOW_AUTOSIZE);
  

  clock_t t;
  for (int i = startFrame; i < endFrame; i += skip) {

    t = clock();
    
    if (i > startFrame) {
      cout << "calculating frame " << i << endl;
      updateModels(video[i], &models, minArea, maxArea, true);
    }

    vector< vector<Point> > contours;

    filterAndFindContours(video[i], &contours);
    
    Mat dst = Mat::zeros(video[i].size(), CV_8UC3);
    Mat contour_img = Mat::zeros(video[i].size(), CV_8UC3);

    drawContoursBoxed(contour_img, &contours, minArea, maxArea);

    for (int n = 0; n < models.size(); n++) {
      int t = models[n].curTime();
      models[n].drawContour(dst, t);
      models[n].drawBoundingBox(dst, Scalar(255, 0, 0), t);
      models[n].drawModel(dst, t);
    }
    
    vector<double> sig = models[0].getRotationSignal();
    Mat h = makeHistImg(sig, 360-models[0].getRotation());

    t = clock() - t;
    printf("It took %f seconds to calculate that frame\n", ((float)t)/CLOCKS_PER_SEC);

    
    Mat dst_fin0, dst_fin1, dst_fin2, filtered;
    Mat frame_copy = video[i].clone();
    drawModels(frame_copy, models, -1);
    combineHorizontal(dst_fin0, dst, frame_copy);
    filter(filtered, video[i]);
    Mat filtered_color;
    cvtColor(filtered, filtered_color, CV_GRAY2RGB);
    combineHorizontal(dst_fin2, filtered_color, contour_img);
    combineVertical(dst_fin1, dst_fin0, dst_fin2);
    Mat dst_fin;
    resize(dst_fin1, dst_fin, Size(), 0.75, 0.75, CV_INTER_AREA);
    out.push_back(dst_fin);
    
  }
  
  if (write) { writeVideo(output_folder, out, fps); }
  if (write_hq) {
    mkdir(video_folder, 0755);
    char prefix[50];
    sprintf(prefix, "%s/frame", video_folder);
    cout << "Writing hw to " << prefix << endl;
    for (int i = 0; i < out.size(); i++) {
      save_frame_safe(out[i], prefix, ".png");
    }
  }
  createTrackbar("Scrubbing", window, &pos, out.size()-1, scrub);

  
  cout << "size: " << out.size()-1 <<  " val: " << getTrackbarPos("Scrubbing", window) << endl;
  scrub(0, 0);

  waitKey(0);
}



void scrub (int , void* ) {
  imshow(window, out[pos]);
  cout << "Frame: " << pos << endl;
  for (int i = 0; i < models.size(); i++) {
    cout << models[i].get_info_string(pos) << endl;
  }
}

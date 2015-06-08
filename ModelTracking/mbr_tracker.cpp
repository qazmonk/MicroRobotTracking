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
int fps, ex;
Size S;
int time_idx = 0, minArea = -1, maxArea = -1;
Mat current_frame;
pthread_mutex_t frame_mutex, output_mutex;
bool new_frame = false, new_output = false, kill_proccessing = false, capturing = false;
vector<Model> models;
vector<bool> selected;
Size frame_size;

int capture(Mat * dst) {
  usleep(250000);
  time_idx++;
  if (time_idx > video.size()) {
    return 1;
  }
  cout << "captured " << (time_idx - 1) << endl;
  *dst = video[time_idx-1];
  return 0;
}

int (*cap)(Mat * dst) = *capture;

void* capture_input(void*) {
  capturing = true;
  while(true) {
    Mat m;
    int rc = (*cap)(&m);
    if (rc) {
      cout << "Capture function signified end of data...exiting" << endl;
      capturing = false;
      pthread_exit(NULL);
    }
    pthread_mutex_lock(&frame_mutex);
    current_frame = m;
    new_frame = true;
    pthread_mutex_unlock(&frame_mutex);
  }
}
void process_output() {
  Mat dst = Mat::zeros(frame_size, CV_8UC3);
  for (int i = 0; i < models.size(); i++) {
    cout << "Drawing model " << i << endl;
    models[i].drawContour(dst, models[i].curTime());
    models[i].drawMask(dst, models[i].curTime());
  }
  cout << "Displaying image" << endl;
  imshow("Output", dst);
  waitKey(10);
}
void* process_input(void*) {
  while(true) {
    if (new_frame) {
      cout << "waiting to copy frame" << endl;
      pthread_mutex_lock(&frame_mutex);
      Mat frame = current_frame;
      new_frame = false;
      pthread_mutex_unlock(&frame_mutex);
      updateModels(frame, &models, minArea, maxArea);
      process_output();
    }
    if (kill_proccessing) {
      cout << "killing processing" << endl;
      pthread_exit(NULL);
    }
  }
}

int main(int argc, char* argv[]) {
  captureVideo(argv[1], &video, &fps, &S, &ex);
  for (int i = 0; i < argc; i++) {
    if (strncmp(argv[i], "--bounds", 10) == 0) {
      minArea = stoi(argv[i+1]);
      maxArea = stoi(argv[i+2]);
      i+=2;
    }
  }
  Mat frame0;
  capture(&frame0);
  current_frame = frame0;
  frame_size = frame0.size();
  new_frame = true;
  //vector<Point> pts = getAffineTransformPoints(frame0, *capture, 640, 480, 650, 10);
  if (minArea < 0 || maxArea < 0 || minArea > maxArea) {
    Point minMax = getMinAndMaxAreas(frame0);
    minArea = minMax.x;
    maxArea = minMax.y;
  }
  Mat frame1;
  capture(&frame1);
  generateModels(frame1, &models, minArea, maxArea);
  pthread_t cap_thread, proc_thread;
  namedWindow("Output", CV_WINDOW_AUTOSIZE);


  /* START CAPTURING AND PROCESSING INPUT */
  pthread_create(&cap_thread, NULL, capture_input, NULL);
  pthread_create(&proc_thread, NULL, process_input, NULL);

  /* PERFORM USER INPUT CONCURRENTLY WITH CAPTURING AND PROCESSING */
  vector<Model> models_copy = models;
  vector<int> selected_ids =  mtlib::selectObjects(frame0, &models_copy);
  cout << "making selected_models" << endl;
  vector<Model> selected_models;
  vector<int> selectidx_to_modelsidx;
  for (int i = 0; i < selected_ids.size(); i++) {
    selected_models.push_back(models_copy[selected_ids[i]]);
    selectidx_to_modelsidx.push_back(selected_ids[i]);
  }
  cout << "selecting masks" << endl;
  cout << selected_models.size() << endl;;
  selectMasks(frame0, &selected_models);

  
  /* KILL PROCESSING AND MOVE DATA AROUND TO AVOID RACE CONDITINOS */
  cout << "killing processing" << endl;
  kill_proccessing = true;  
  void * status;
  int rc = pthread_join(proc_thread, &status);
  if (rc) { 
    printf("ERROR: return code from proc_thread is %d\n", rc);
    exit(-1);
  }
  vector<Model> selected_models2;
  for (int i = 0; i < selected_ids.size(); i++) {
    selected_models2.push_back(models[selected_ids[i]]);
    selected_models2.back().mask = selected_models[i].mask;
  }
  models = selected_models2;

  /* RESTART PROCESSING */
  cout << "restarting processing" << endl;
  kill_proccessing = false;
  pthread_create(&proc_thread, NULL, process_input, NULL);
  cout << "restarted processing" << endl;
  /* WAIT FOR CAPTURIG TO FINISH */
  while(capturing) {waitKey(0);}
  rc = pthread_join(cap_thread, &status);
  if (rc) { 
    printf("ERROR: return code from cap_thread is %d\n", rc);
    exit(-1);
  }
 
  cout << "Input finished...exiting" << endl;
  pthread_exit(NULL);
  return 0;
}

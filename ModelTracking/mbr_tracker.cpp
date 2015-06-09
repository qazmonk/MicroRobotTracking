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
Mat current_frame, toggle_on, toggle_off, current_cont_frame;
pthread_mutex_t frame_mutex, output_mutex, flag_mutex, models_mutex, cont_frame_mutex;
bool new_frame = false, new_output = false, kill_proccessing = false, capturing = false;
bool outputting = false;
vector<Model> models;
vector<bool> selected, exposing;
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
  toggle_on = imread("toggle_on.png");
  toggle_off = imread("toggle_on.png");
  while(true) {
    Mat m;
    int rc = (*cap)(&m);
    if (rc) {
      cout << "Capture function signified end of data...exiting" << endl;
      pthread_mutex_lock(&flag_mutex);
      capturing = false;
      pthread_mutex_unlock(&flag_mutex);
      pthread_exit(NULL);
    }
    pthread_mutex_lock(&frame_mutex);
    current_frame = m;
    new_frame = true;
    pthread_mutex_unlock(&frame_mutex);
  }
}
bool lastMouseButton = false;

void selectExposuresCallback(int event, int x, int y, int, void*) {
  if (event != EVENT_LBUTTONDOWN) {
    lastMouseButton = false;
    return;
  }

  if (lastMouseButton == false) {
    pthread_mutex_lock(&cont_frame_mutex);
    cv::Mat frame = current_cont_frame.clone();
    pthread_mutex_unlock(&cont_frame_mutex);

    pthread_mutex_lock(&models_mutex);
    vector<mtlib::Model> models_copy = models;
    vector<bool> exposing_copy = exposing;
    pthread_mutex_unlock(&models_mutex);
    double min_area = -1;
    int min = -1;
    for (int i = 0; i < models_copy.size(); i++) {
      if (pointInRotatedRectangle(x, y, models_copy.at(i).getBoundingBox())
          && (min_area < 0 || models_copy.at(i).getArea() <= min_area)) {
        min = i;
        min_area = models_copy.at(i).getArea();
      }
    }
    if (min != -1) {
      exposing_copy[min] = !exposing_copy[min];
      pthread_mutex_lock(&cont_frame_mutex);
      exposing[min] = exposing_copy[min];
      pthread_mutex_unlock(&cont_frame_mutex);
    }

    for (int n = 0; n < models_copy.size(); n++) {
      Scalar color(0, 0, 255);
      if (exposing_copy[n]) {
        color = Scalar(0, 255, 0);
      }
      Point2f verticies[4];
      models_copy.at(n).getBoundingBox().points(verticies);

      for (int i = 0; i < 4; i++)
        line(frame, verticies[i], verticies[(i+1)%4], color, 2);

    }
    imshow("Tracking", frame);
  }
  lastMouseButton = true;
}
void process_output() {
  Mat dst = Mat::zeros(frame_size, CV_8UC3);
  Mat dst2 = Mat::zeros(frame_size, CV_8UC3);
  for (int i = 0; i < models.size(); i++) {
    cout << "Drawing model " << i << endl;
    if (exposing[i]) {
      models[i].drawContour(dst, models[i].curTime());
      models[i].drawMask(dst, models[i].curTime());
    }

    models[i].drawContour(dst2, models[i].curTime());
    Scalar color(0, 0, 255);
    if (exposing[i]) {
      color = Scalar(0, 255, 0);
    }
    Point2f verticies[4];
    models[i].getBoundingBox().points(verticies);
    for (int j = 0; j < 4; j++)
      line(dst2, verticies[j], verticies[(j+1)%4], color, 2);
  }
  current_cont_frame = dst2;
  cout << "Displaying image" << endl;
  imshow("Output", dst);
  imshow("Tracking", dst2);
  waitKey(10);
}
void* process_input(void*) {
  bool new_frame_copy = false, outputting_copy;
  Mat frame;
  while(true) {
    /* copy the state of the flags */
    pthread_mutex_lock(&flag_mutex);
    if (kill_proccessing) {
      cout << "killing processing" << endl;
      pthread_exit(NULL);
    }
    outputting_copy = outputting;
    pthread_mutex_unlock(&flag_mutex);

    /* check for a new frame and copy it if it exists */
    new_frame_copy = false;
    pthread_mutex_lock(&frame_mutex);
    new_frame_copy = new_frame;
    if (new_frame) {
      frame = current_frame;
      new_frame = false;
    }
    pthread_mutex_unlock(&frame_mutex);

    /* if a new frame was found process it */
    if (new_frame_copy) {
      pthread_mutex_lock(&models_mutex);
      updateModels(frame, &models, minArea, maxArea);
      if (outputting_copy) {
        process_output();
      }
      pthread_mutex_unlock(&models_mutex);
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
  outputting = false;
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
  namedWindow("Tracking", CV_WINDOW_AUTOSIZE);
  setMouseCallback("Tracking", selectExposuresCallback, 0);
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
  pthread_mutex_lock(&flag_mutex);
  kill_proccessing = true;  
  pthread_mutex_unlock(&flag_mutex);

  void * status;
  int rc = pthread_join(proc_thread, &status);
  if (rc) { 
    printf("ERROR: return code from proc_thread is %d\n", rc);
    exit(-1);
  }
  vector<Model> selected_models2;
  for (int i = 0; i < selected_ids.size(); i++) {
    selected_models2.push_back(models[selected_ids[i]]);
    selected_models2.back().setMask(selected_models[i].getMask());
  }
  models = selected_models2;
  exposing = vector<bool>(models.size(), false);
  /* RESTART PROCESSING */
  cout << "restarting processing" << endl;
  kill_proccessing = false;
  outputting = true;
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
  return 0;
}

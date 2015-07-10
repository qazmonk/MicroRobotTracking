#include <chrono>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdlib.h>
#include <unistd.h>
#include "mtlib.h"
#include <algorithm>
#include <time.h>
#include "firefly.h" 
#include <sys/stat.h>
#include <sstream>

using namespace std;
using namespace cv;
using namespace mtlib;
using std::setw;

vector<Mat> video;
int fps = 15, ex, frame_count = 0;
Size S, input_size, dmd_size;
int time_idx = 0, minArea = -1, maxArea = -1, dmd_w, dmd_h;
Mat current_frame, toggle_on, toggle_off, current_cont_frame, warp_mat;
unsigned long current_time=0;
pthread_mutex_t frame_mutex, output_mutex, flag_mutex, models_mutex, cont_frame_mutex,
  frame_count_mutex;
bool new_frame = false, new_output = false, kill_proccessing = false, capturing = false;
bool outputting = false, writing = false;;
vector<Model> models;
vector<bool> selected, exposing;
Size frame_size;
firefly_t * f;
firefly_camera_t * camera;
vector<Mat> output;
bool masking = false;
bool perspective = false;
VideoWriter output_cap;
const char * output_file_prefix, *output_file_suffix;

unsigned long time_milliseconds() {
  return chrono::duration_cast<chrono::milliseconds>
    (chrono::system_clock::now().time_since_epoch()).count();
}
void dmd_imshow(Mat im) {
  
  Mat inv;
  bitwise_not(im, inv);
  Mat warped(Size(dmd_w, dmd_h), CV_8UC3);
  Mat expanded = expandForDMD(inv, dmd_w, dmd_h);
  warpPerspective(expanded, warped, warp_mat, warped.size(),
                  INTER_LINEAR, BORDER_CONSTANT, Scalar(255, 255, 255));
  imshow("DMD", warped);
}
int capture_from_video(Mat * dst) {
  usleep(30000);
  time_idx++;
  if (time_idx > video.size()) {
    return -1;
  }
  cout << "captured " << (time_idx - 1) << endl;
  *dst = video[time_idx-1];
  return 0;
}
int capture_from_camera(Mat * dst) {
  int rc = opencv_firefly_capture(camera, dst);
  //bitwise_not(*dst, *dst);

  return rc;
}
int (*cap)(Mat * dst) = *capture_from_video;

void* capture_input(void*) {
  capturing = true;
  unsigned long time = time_milliseconds();
  while(capturing) {
    time = time_milliseconds();
    Mat m;
    int t = (*cap)(&m);
    if (t < 0) {
      cout << "Capture function signified end of data...exiting" << endl;
      pthread_mutex_lock(&flag_mutex);
      capturing = false;
      pthread_mutex_unlock(&flag_mutex);
      pthread_exit(NULL);
    }
    pthread_mutex_lock(&frame_mutex);
    current_frame = m.clone();
    new_frame = true;
    current_time = t;
    pthread_mutex_unlock(&frame_mutex);
    pthread_mutex_lock(&frame_count_mutex);
    time = time_milliseconds() - time;
    //printf("%lu milliseconds to capture frame %d\n", time, frame_count);
    pthread_mutex_unlock(&frame_count_mutex);
  }
  return NULL;
}
bool lastMouseButton = false;

void selectExposuresCallback(int event, int x, int y, int, void*) {
  if (event != EVENT_LBUTTONDOWN) {
    lastMouseButton = false;
    return;
  }

  if (lastMouseButton == false) {
    /*pthread_mutex_lock(&cont_frame_mutex);
    cv::Mat frame = current_cont_frame.clone();
    pthread_mutex_unlock(&cont_frame_mutex);*/
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
      pthread_mutex_lock(&models_mutex);
      exposing[min] = exposing_copy[min];
      pthread_mutex_unlock(&models_mutex);
    }
    /*for (int n = 0; n < models_copy.size(); n++) {
      Scalar color(0, 0, 255);
      if (exposing_copy[n]) {
        color = Scalar(0, 255, 0);
      }
      Point2f verticies[4];
      models_copy.at(n).getBoundingBox().points(verticies);

      for (int i = 0; i < 4; i++)
        line(frame, verticies[i], verticies[(i+1)%4], color, 2);

    }*/
  }
  lastMouseButton = true;
}
void process_output() {
  Mat dst = Mat::zeros(frame_size, CV_8UC3);
  Mat dst2 = Mat::zeros(frame_size, CV_8UC3);
  unsigned long t;
  for (int i = 0; i < models.size(); i++) {
    t = time_milliseconds();
    if (exposing[i]) {
      if (masking) {
        models[i].drawContour(dst, models[i].curTime());
        models[i].drawMask(dst, models[i].curTime());
      } else {
        models[i].drawExposure(dst, models[i].curTime());
      }

    }
    t = time_milliseconds() - t;
    //printf("%lu milliseconds to draw models for dmd\n", t);
    t = time_milliseconds();
    models[i].drawContour(dst2, models[i].curTime());
    Scalar color(0, 0, 255);
    if (exposing[i]) {
      color = Scalar(0, 255, 0);
    }
    Point2f verticies[4];
    models[i].getBoundingBox().points(verticies);
    for (int j = 0; j < 4; j++)
      line(dst2, verticies[j], verticies[(j+1)%4], color, 2);
    t = time_milliseconds() - t;
    //printf("%lu milliseconds to draw models for Tracking window\n", t);
  }
  /*pthread_mutex_lock(&cont_frame_mutex);
  current_cont_frame = dst2.clone();
  pthread_mutex_unlock(&cont_frame_mutex);*/
  t = time_milliseconds();
  dmd_imshow(dst);
  imshow("Tracking", dst2);
  t = time_milliseconds() - t;
  //printf("%lu milliseconds to display images\n", t);
}
void new_file(int n) {
  char buff[50];
  sprintf(buff, "%s-%.4d%s", output_file_prefix, n, output_file_suffix);
  output_cap = VideoWriter(buff, 
                           CV_FOURCC('m', 'p', '4', 'v'),
                           30,
                           input_size);
}
void* process_input(void*) {
  bool new_frame_copy = false, outputting_copy;
  Mat frame;
  unsigned long timestamp;
  unsigned long t, t1, t2, t3;
  while(true) {
    /* copy the state of the flags */
    t = time_milliseconds();
    t1 = time_milliseconds();
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
      frame = current_frame.clone();
      timestamp = current_time;
      new_frame = false;
    }
    pthread_mutex_unlock(&frame_mutex);
    t1 = time_milliseconds() - t1;
    pthread_mutex_lock(&frame_count_mutex);
    //printf("%lu milliseconds to copy data for frame %d\n", t1, frame_count);
    pthread_mutex_unlock(&frame_count_mutex);
    /* if a new frame was found process it */
    if (new_frame_copy) {

      pthread_mutex_lock(&models_mutex);
      updateModels(frame.clone(), &models, minArea, maxArea, timestamp);
      t2 = time_milliseconds();
      imshow("Camera", frame);
      if (outputting_copy) {
        process_output();
      }
      t2 = time_milliseconds() - t2;
      pthread_mutex_unlock(&models_mutex);
      t3 = time_milliseconds();
      if (outputting_copy && writing) {
        if (frame_count%1000 == 0 && frame_count > 0) {
          new_file(frame_count/1000);
        }
        output_cap.write(frame);
        pthread_mutex_lock(&frame_count_mutex);
        frame_count++;
        pthread_mutex_unlock(&frame_count_mutex);
      }
      t3 = time_milliseconds() - t3;
      t = time_milliseconds() - t;

      pthread_mutex_lock(&frame_count_mutex);
      /*printf("%lu milliseconds to write frame %d\n", t3, 
             frame_count);
      printf("%lu milliseconds to process output for frame %d\n", t2, 
             frame_count);
      printf("%lu milliseconds to process frame %d\n", t, 
      frame_count);*/
      pthread_mutex_unlock(&frame_count_mutex);
    }
  }
}

int main(int argc, char* argv[]) {
  Model::init();
  bool using_camera = false, input_given = false, exposure_given = false;
  bool writing_data = false;
  char * output_file, *data_file;
  int  dmd_x = 39, dmd_y = 1400;
  dmd_w = 608;
  dmd_h = 662;
  for (int i = 0; i < argc; i++) {
    if (strncmp(argv[i], "--bounds", 10) == 0) {
      minArea = stoi(argv[i+1]);
      maxArea = stoi(argv[i+2]);
      i+=2;
    } else if (strncmp(argv[i], "--camera", 10) == 0) {
      using_camera = true;
      cap = *capture_from_camera;
      f = firefly_new();
      firefly_setup_camera(f, &camera);
      firefly_start_transmission(camera);
      input_given = true;
    } else if (strncmp(argv[i], "--write-video", 10) == 0) {
      writing = true;
      output_file_prefix = argv[i+1];
      output_file_suffix = ".mp4";
      i++;
    } else if (strncmp(argv[i], "--write-data", 10) == 0) {
      writing_data = true;
      data_file = argv[i+1];
      i++;
    } else if (strncmp(argv[i], "--expose", 10) == 0) {
      exposure_given = true;
      masking = false;
    } else if (strncmp(argv[i], "--mask", 10) == 0) {
      exposure_given = true;
      masking = true;
    } else if (strncmp(argv[i], "--file", 10) == 0) {
      using_camera = false;
      input_given = true;
      captureVideo(argv[i+1], &video, &fps, &S, &ex);      
      i++;
    } else if (strncmp(argv[i], "--dmd-args", 15) == 0) {
      dmd_w = stoi(argv[i+1]);
      dmd_h = stoi(argv[i+2]);
      dmd_x = stoi(argv[i+3]);
      dmd_y = stoi(argv[i+4]);
      i += 4;
    } else if (strncmp(argv[i], "--help", 15) == 0) {
      cout << "This is mbr_tracker, a utility to track microstructures"
           << " in the MSRL and Lynch laboratories" << endl << endl;
      cout << "\tThere are two arguments that must be given:" << endl
           << "--expose/--mask, and --camera/--file. The first pair chooses" << endl
           << "whether you want to expose predified shapes on top of the MBRs" << endl
           << "or mask their contours. The second specifies camera or file" << endl
           << "input, the file input must be followed by the name of a video" << endl
           << "file." << endl;
      cout << "Other flags are:" << endl;
      cout << "--bounds <integer> <integer>" << endl << right << setw(90)
           << "Prechooses the area bounds for contour"
        "filtering" << endl;
      cout << "--write-video <filename>" << endl << right << setw(90)
           << "Writes the input video to the specified filename"
           << endl; 
      cout << "--write-data <filename>" << endl << setw(90)
           << "Writes the data"
        " of the tracked model to the specified filename" << endl;
      cout <<"--dmd-args <integer> <integer> <integer> <integer>" << endl << setw(90)
           << " Sets the width, height, x position, and y position" 
           " of the output window" << endl;
      return 0;
    }
  }
  if (!writing_data) {
    writing_data = true;
    time_t rawtime;
    struct tm * timeinfo;
    char buff[80];
    
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(buff,80,"tracking_output-%m-%d-%Y-%H:%M", timeinfo);
    data_file = buff;
  }
  if (!writing) {
    cout << "Are you sure you don't want to save a video? (y/n): " << endl;
    char rsp;
    cin >> rsp;
    if (rsp != 'y') {
      cout << "Restart the program with the '--write-video <filename>' flag" << endl;
      exit(0);
    }
  }
  dmd_size = Size(dmd_w, dmd_h);
  if (!input_given) {
    cout << "Must specify input using '--file' or '--camera'" << endl;
    return 0;
  }
  if (!exposure_given) {
    cout << "Must specify exposure type using '--expose' or '--mask'" << endl;
    return 0;
  }
  Mat frame0;
  new_frame = true;
  outputting = false;
   
  namedWindow("DMD", CV_WINDOW_NORMAL);
  cvMoveWindow("DMD", dmd_x, dmd_y);
  cvResizeWindow("DMD", dmd_w, dmd_h);

  cap(&frame0);
  input_size = frame0.size();
  current_frame = frame0;
  frame_size = frame0.size();
  bool using_file_transform = false;
  string warp_file = "warp_data.data";
  Point2f src_pts[4];
  Point2f dst_pts[4];
  int num_points = 4;
  if (file_exists(warp_file)) {
    ifstream warp_file_stream;
    warp_file_stream.open(warp_file, ios::in);
    string line;
    getline(warp_file_stream, line);
    istringstream src_in(line);
    getline(warp_file_stream, line);
    istringstream dst_in(line);
    float x, y;
    for (int i = 0; i < num_points; i++) {
      src_in >> x >> y;
      src_pts[i] = Point2f(x, y);
    }
    for (int i = 0; i < num_points; i++) {
      dst_in >> x >> y;
      dst_pts[i] = Point2f(x, y);
    }
    warp_mat = getPerspectiveTransform(dst_pts, src_pts);
    Mat checkerboard(dmd_size, CV_8UC3, Scalar(255, 255, 255));
    Size square(dmd_size.width/4, dmd_size.height/4);
    for (int i = 0; i < 4; i += 1) {
      for (int j = 0; j < 4; j += 1) {
        if ((i+j)%2 == 0) {
          Rect r(Point(i*square.width, j*square.height), square);
          rectangle(checkerboard, r, Scalar(0, 0, 0), CV_FILLED);
        }
      }
    }
    Mat warped(dmd_size, CV_8UC3);
    warpPerspective(checkerboard, warped, warp_mat, warped.size(),
                    INTER_LINEAR, BORDER_CONSTANT, Scalar(255, 255, 255));
    dmd_imshow(warped);

    cout << "Press 'y' to use the saved transform" << endl;
    namedWindow("Saved Transform", CV_WINDOW_AUTOSIZE);
    while (true) {
      Mat tmp;
      cap(&tmp);
      imshow("Saved Transform", tmp);
      int code = waitKey(1000/60);
      if (code == 'y') {
        using_file_transform = true;
        break;
      } else if (code != -1) {
        break;
      }
    }
  }
  if (!using_file_transform) {
    vector<Point2f> pts;
    pts = autoCalibrate(cap, "DMD", Size(dmd_w, dmd_h));
    for (int i = 0; i < num_points; i++) src_pts[i] = pts[i];
    for (int i = 0; i < num_points; i++) dst_pts[i] = pts[i+num_points];
    ofstream warp_file_stream;
    warp_file_stream.open(warp_file, ios::out);
    
    for (int i = 0; i < num_points; i++) {
      warp_file_stream << src_pts[i].x << " " << src_pts[i].y << " ";
    }
    warp_file_stream << endl;
    for (int i = 0; i < num_points; i++) {
      warp_file_stream << dst_pts[i].x << " " << dst_pts[i].y << " ";
    }
  }
  warp_mat = getPerspectiveTransform(dst_pts, src_pts);
  cout << warp_mat << endl;
  if (minArea < 0 || maxArea < 0 || minArea > maxArea) {
    Point minMax = getMinAndMaxAreas(frame0);
    minArea = minMax.x;
    maxArea = minMax.y;
  }
  Mat black(Size(dmd_w, dmd_h), CV_8UC3, Scalar(0, 0, 0));
  dmd_imshow(black);
  usleep(1000);
  cap(&frame0);
  while (true) {
    models.clear();
    generateModels(frame0, &models, minArea, maxArea);
    Mat tmp = Mat::zeros(frame0.size(), frame0.type());
    for (int i = 0; i < models.size(); i++) {
      models[i].drawContour(tmp, 0);
      models[i].drawBoundingBox(tmp, 0, Scalar(255, 0, 0));
    }
    namedWindow("Models", CV_WINDOW_AUTOSIZE);
    imshow("Models", tmp);
    cout << "Would you like to use a different image for model selection? (y/n)" << endl;
    char ans;
    cin >> ans;
    if (ans == 'y' || ans == 'Y') { 
      cout << "Restarting model generation" << endl;
      firefly_flush_camera_no_restart(camera);
      cap(&frame0);
    } else {
      destroyWindow("Models");
      break;
    }
    destroyWindow("Models");
  }
  pthread_t cap_thread, proc_thread;
 
  dmd_imshow(black);
  namedWindow("Tracking", CV_WINDOW_AUTOSIZE);
  namedWindow("Camera", CV_WINDOW_AUTOSIZE);
  setMouseCallback("Tracking", selectExposuresCallback, 0);
  /* START CAPTURING AND PROCESSING INPUT */
  pthread_create(&cap_thread, NULL, capture_input, NULL);
  pthread_create(&proc_thread, NULL, process_input, NULL);

  /* PERFORM USER INPUT CONCURRENTLY WITH CAPTURING AND PROCESSING */
  vector<Model> models_copy = models;
  vector<int> selected_ids =  mtlib::selectObjects(frame0, &models_copy);
  vector<Model> selected_models;
  vector<int> selectidx_to_modelsidx;
  for (int i = 0; i < selected_ids.size(); i++) {
    selected_models.push_back(models_copy[selected_ids[i]]);
    selectidx_to_modelsidx.push_back(selected_ids[i]);
  }
  if (selected_ids.size() < 1) {
    cout << "No models selected. Exiting." << endl;
    return 0;
  }
  if (masking) {
    selectMasks(frame0, &selected_models);
  } else {
    selectExposures(frame0, &selected_models);
  }

  
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
    selected_models2.back().setExposure(selected_models[i].getExposure());
  }
  models = selected_models2;
  exposing = vector<bool>(models.size(), false);
  /* RESTART PROCESSING */
  cout << "restarting processing" << endl;
  if (writing) {
    new_file(0);
  }
  
  kill_proccessing = false;
  outputting = true;
  pthread_create(&proc_thread, NULL, process_input, NULL);
  cout << "restarted processing" << endl;
  /* WAIT FOR CAPTURIG TO FINISH */
  while(capturing) {
    if (waitKey(0) == 'x') capturing = false;
  }
  rc = pthread_join(cap_thread, &status);
  if (rc) { 
    printf("ERROR: return code from cap_thread is %d\n", rc);
    exit(-1);
  }
  if (writing_data) {
    string data_name = safe_filename(data_file, ".ssv");
    cout << "Writing data to " << data_name << endl;
    
    ofstream data_file_stream;
    data_file_stream.open(data_name, ios::out);
    for (int n = 0; n < models.size(); n++) {
      models[n].write_data(&data_file_stream);
    }
  }
  cout << "Input finished...exiting" << endl;
  if (writing) {
    output_cap.release();
  }
  return 0;
}

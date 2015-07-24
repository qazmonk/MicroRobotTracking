#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "firefly.h"
#include "mtlib.h"
#include <sys/stat.h>
#include <iostream>
#include <iomanip>
#include <fstream>
using namespace std;
using namespace cv;

firefly_t * f;
firefly_camera_t * camera;

long capture_from_camera(Mat * dst) {
  long rc = opencv_firefly_capture(camera, dst);
  //bitwise_not(*dst, *dst);

  return rc;
}
int main(int argc, char* argv[]) {
  mtlib::Model::init();
  bool writing = false, displaying = false;
  char * output_folder;
  Mat display_image, warp_mat;
  int  dmd_x = 39, dmd_y = 1400;
  int dmd_w = 608;
  int dmd_h = 662;

  for (int i = 0; i < argc; i++) {
    if (strncmp(argv[i], "--write", 10) == 0) {
      writing = true;
      cout << "writing turned on" << endl;
      output_folder = argv[i+1];
      i++;
    }
    else if (strncmp(argv[i], "--display", 11) == 0) {
      displaying = true;
      display_image = imread(argv[i+1], CV_LOAD_IMAGE_GRAYSCALE);
      i++;
    }
    else if (strncmp(argv[i], "--auto-calibrate", 30) == 0) {
      vector<Point2f> pts;
      vector<Point2f> src_pts, dst_pts;
      pts = mtlib::autoCalibrate(capture_from_camera, "DMD", Size(dmd_w, dmd_h));
      for (int i = 0; i < 4; i++) src_pts[i] = pts[i];
      for (int i = 0; i < 4; i++) dst_pts[i] = pts[i+4];
      warp_mat = getPerspectiveTransform(dst_pts, src_pts);
    } else if (strncmp(argv[i], "--file-calibrate", 30) == 0) {
      Point2f src_pts[4], dst_pts[4];
      ifstream warp_file_stream;
      warp_file_stream.open(argv[i+1], ios::in);
      string line;
      getline(warp_file_stream, line);
      istringstream src_in(line);
      getline(warp_file_stream, line);
      istringstream dst_in(line);
      float x, y;
      for (int i = 0; i < 4; i++) {
        src_in >> x >> y;
        src_pts[i] = Point2f(x, y);
      }
      for (int i = 0; i < 4; i++) {
        dst_in >> x >> y;
        dst_pts[i] = Point2f(x, y);
      }
      warp_mat = getPerspectiveTransform(dst_pts, src_pts);
      i++;
    }
  }
  if (displaying) {
    namedWindow("DMD", CV_WINDOW_NORMAL);
    cvMoveWindow("DMD", dmd_x, dmd_y);
    cvResizeWindow("DMD", dmd_w, dmd_h);

    Mat warped(Size(dmd_w, dmd_h), CV_8UC3);  
    warpPerspective(display_image, warped, warp_mat, warped.size(),
                    INTER_LINEAR, BORDER_CONSTANT, Scalar(255, 255, 255));


    imshow("DMD", warped);
  }
  f = firefly_new();


  firefly_setup_camera(f, &camera);

  Mat frame;
  namedWindow("Frame", CV_WINDOW_AUTOSIZE);
  firefly_start_transmission(camera);
  opencv_firefly_capture(camera, &frame);
  //firefly_get_color_transform(camera);

  vector<Mat> video;
  unsigned long numFrames = 0;
  double average = 0;
  unsigned long last_timestamp = opencv_firefly_capture(camera, &frame);
  while (waitKey(1000/45) == -1) {
    unsigned long time = opencv_firefly_capture(camera, &frame);
    imshow("Frame", frame);
    numFrames++;
    unsigned long delay = time - last_timestamp;
    average = delay/((double)numFrames) + (numFrames - 1)*average/((double)numFrames);
    last_timestamp = time;
    if (writing) {
      video.push_back(frame);
    }
  }
  cout << "average delay: " << average << endl;
  int fps = 1000/average;
  cout << "fps: " << fps << endl;
  if (writing) {
    cout << "Writing video" << endl;
    /*bool succ = mtlib::writeVideo(output_file, video, fps);
    if (!succ) {
      cout << "Error writing video" << endl;
    }*/
    mkdir(output_folder, 0755);
    char prefix[50];
    sprintf(prefix, "%s/frame", output_folder);
    cout << "Writing hq to " << prefix << endl;
    for (int i = 0; i < video.size(); i++) {
      mtlib::save_frame_safe(video[i], prefix, ".png");
    }
  }
  firefly_stop_transmission(camera);
  firefly_cleanup_camera(camera);
  firefly_free(f);
  
  return 0;
}

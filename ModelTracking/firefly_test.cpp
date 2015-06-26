#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "firefly.h"
#include "mtlib.h"
using namespace std;
using namespace cv;


int main(int argc, char* argv[]) {
  mtlib::Model::init();
  bool writing = false;
  char * output_file;
  for (int i = 0; i < argc; i++) {
    if (strncmp(argv[i], "--write", 10) == 0) {
      writing = true;
      cout << "writing turned on" << endl;
      output_file = argv[i+1];
      i++;
    }
  }
  firefly_t * f = firefly_new();
  firefly_camera_t * camera;

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
  while (waitKey(1000/60) == -1) {
    unsigned long time = opencv_firefly_capture(camera, &frame);
    imshow("Frame", frame);
    numFrames++;
    unsigned long delay = time - last_timestamp;
    average = delay/((double)numFrames) + (numFrames - 1)*average/((double)numFrames);
    last_timestamp = time;
  }
  cout << "average delay: " << average << endl;
  int fps = 1000/average;
  cout << "fps: " << fps << endl;
  if (writing) {
    cout << "Writing video" << endl;
    bool succ = mtlib::writeVideo(output_file, video, fps);
    if (!succ) {
      cout << "Error writing video" << endl;
    }
  }
  firefly_stop_transmission(camera);
  firefly_cleanup_camera(camera);
  firefly_free(f);
  
  return 0;
}

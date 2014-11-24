#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "firefly.h"
using namespace std;
using namespace cv;

int main() {
  firefly_t * f = firefly_new();
  firefly_camera_t * camera;
  
  firefly_setup_camera(f, &camera);

  firefly_start_transmission(camera);

  Mat frame = firefly_capture_frame(camera);

  firefly_stop_transmission(camera);

  namedWindow("Frame", CV_WINDOW_AUTOSIZE);
  imshow("Frame", frame);
  waitKey(0);

  firefly_cleanup_camera(camera);
  firefly_free(f);

  return 0;
}

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "firefly.h"
using namespace std;
using namespace cv;

int main() {
  firefly_t * f = firefly_new();
  firefly_camera_t * camera;
  firefly_frame_t frame;

  firefly_setup_camera(f, &camera);

  namedWindow("Frame", CV_WINDOW_AUTOSIZE);
  firefly_start_transmission(camera);

  while (waitKey(1000/60) == -1) {
    frame = firefly_capture_frame(camera);
    imshow("Frame", frame.img);
    if (frame.frames_behind > 0) {
      firefly_flush_camera(camera);
      firefly_start_transmission(camera);
    }
  }
  firefly_stop_transmission(camera);
  firefly_cleanup_camera(camera);
  firefly_free(f);

  return 0;
}

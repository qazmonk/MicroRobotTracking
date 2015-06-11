#ifndef _FIREFLY_H
#define _FIREFLY_H
#include <dc1394/dc1394.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

typedef dc1394_t firefly_t;
typedef dc1394camera_t firefly_camera_t;
typedef dc1394error_t firefly_error_t;
typedef struct
{
  cv::Mat img;
  int frames_behind;
  firefly_error_t err;
} firefly_frame_t;

firefly_t * firefly_new();

firefly_error_t firefly_setup_camera(firefly_t * f, firefly_camera_t ** camera);

firefly_error_t firefly_start_transmission(firefly_camera_t * camera);

firefly_frame_t firefly_capture_frame(firefly_camera_t * camera);

firefly_error_t firefly_stop_transmission(firefly_camera_t * camera);

void firefly_flush_camera(firefly_camera_t * camera);

void firefly_cleanup_camera(firefly_camera_t * camera);

void firefly_free(firefly_t * f);



#endif

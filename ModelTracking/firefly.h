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
  unsigned long timestamp;
} firefly_frame_t;

firefly_t * firefly_new();

firefly_error_t firefly_setup_camera(firefly_t * f, firefly_camera_t ** camera);

firefly_error_t firefly_start_transmission(firefly_camera_t * camera);

void firefly_get_color_transform(firefly_camera_t * camera);

firefly_frame_t firefly_capture_frame(firefly_camera_t * camera, int code=CV_BayerBG2BGR);

firefly_error_t firefly_stop_transmission(firefly_camera_t * camera);

void firefly_flush_camera(firefly_camera_t * camera);

void firefly_flush_camera_no_restart(firefly_camera_t * camera);

void firefly_cleanup_camera(firefly_camera_t * camera);

void firefly_free(firefly_t * f);

int opencv_firefly_capture(firefly_camera_t * camera, cv::Mat * dst, int code=CV_BayerBG2BGR);

#endif

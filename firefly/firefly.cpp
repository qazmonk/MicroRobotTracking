
#include <dc1394/dc1394.h>
#include <dc1394/format7.h>
#include <stdlib.h>
#include <inttypes.h>
#include <opencv2/opencv.hpp>


#include "firefly.h"

void cleanup_and_exit(firefly_camera_t *camera)
{
    dc1394_video_set_transmission(camera, DC1394_OFF);
    dc1394_capture_stop(camera);
    dc1394_camera_free(camera);
    exit(1);
}
firefly_t * firefly_new() {
  return dc1394_new();
}
firefly_error_t firefly_setup_camera(firefly_t * f, firefly_camera_t ** camera) {
  int i;
  dc1394featureset_t features;
  dc1394framerates_t framerates;
  dc1394video_modes_t video_modes;
  dc1394framerate_t framerate;
  dc1394video_mode_t video_mode;
  dc1394color_coding_t coding;
  unsigned int width, height;
  dc1394error_t err;
  dc1394camera_list_t * list;

  err=dc1394_camera_enumerate (f, &list);
  if (err < 0) {
    return err;
  }

  if (list->num == 0) {
    dc1394_log_error("No cameras found");
    return DC1394_FAILURE;
  }

  *camera = dc1394_camera_new (f, list->ids[0].guid);
  if (!camera) {
    dc1394_log_error("Failed to initialize camera with guid %"PRIx64, list->ids[0].guid);
    return DC1394_FAILURE;
  }
  dc1394_camera_free_list (list);

  printf("Using camera with GUID %"PRIx64"\n", (*camera)->guid);

  /*-----------------------------------------------------------------------
   *  Set the video mode to format_7 to get color images
   *-----------------------------------------------------------------------*/

  video_mode=DC1394_VIDEO_MODE_FORMAT7_0;
  err=dc1394_get_color_coding_from_video_mode(*camera, video_mode,&coding);

  if (err < 0) {
    printf("Could not get valid color coding\n");
    return err;
  }

  //Get max image size
  unsigned int w, h;
  dc1394_format7_get_max_image_size(*camera, video_mode, &w, &h);
  printf("Maximumg size %dx%d\n", w, h);


  /*-----------------------------------------------------------------------
   *  setup capture
   *-----------------------------------------------------------------------*/

  err=dc1394_video_set_iso_speed(*camera, DC1394_ISO_SPEED_400);
  if (err < 0) {
    printf("Could not set iso speed\n");
    return err;
  }


  err=dc1394_video_set_mode(*camera, video_mode);
  if (err < 0) {
    printf("Could not set video mode\n");
    return err;
  }

  err=dc1394_video_set_framerate(*camera, DC1394_FRAMERATE_60);
  if (err < 0) {
    printf("Could not set framerate\n");
    return err;
  }

  err=dc1394_capture_setup(*camera,4, DC1394_CAPTURE_FLAGS_DEFAULT);
  if (err < 0) {
    printf("Could not setup camera-\nmake sure that the video mode and framerate are\nsupported by your camera\n");
    return err;
  }
  return DC1394_SUCCESS;
}

firefly_error_t firefly_start_transmission(firefly_camera_t * camera) {
  dc1394error_t err = dc1394_video_set_transmission(camera, DC1394_ON);
  if (err < 0) {
    return err;
  }
  return DC1394_SUCCESS;
}

firefly_frame_t firefly_capture_frame(firefly_camera_t * camera) {
  dc1394video_frame_t *frame;
  dc1394error_t err=dc1394_capture_dequeue(camera, DC1394_CAPTURE_POLICY_WAIT, &frame);/* Capture */
  if (err < 0) {
    printf("Problem getting an image\n");
  }

  //Convert to RGB from raw format
  cv::Mat bayer_img(frame->size[1], frame->size[0], CV_8UC1, frame->image, frame->stride);
  cv::Mat img(frame->size[1], frame->size[0], CV_8UC3);
  cv::cvtColor(bayer_img, img, CV_BayerRG2RGB);
  
  dc1394_capture_enqueue(camera, frame);
  firefly_frame_t f;
  f.img = img;
  f.frames_behind = frame->frames_behind;
  return f;
}

void firefly_flush_camera(firefly_camera_t * camera) {
  firefly_stop_transmission(camera);
  dc1394video_frame_t *frame;

  do {
      dc1394error_t err=dc1394_capture_dequeue(camera, DC1394_CAPTURE_POLICY_WAIT, &frame);
      if (err < 0) {
        return;
      }
  } while (frame->frames_behind > 0);
}

firefly_error_t firefly_stop_transmission(firefly_camera_t * camera) {
  dc1394error_t err = dc1394_video_set_transmission(camera, DC1394_OFF);
  if (err < 0) {
    return err;
  }
  return DC1394_SUCCESS;
}

void firefly_cleanup_camera(firefly_camera_t * camera) {
  dc1394_video_set_transmission(camera, DC1394_OFF);
  dc1394_capture_stop(camera);
  dc1394_camera_free(camera);
}

void firefly_free(firefly_t * f) {
  dc1394_free(f);
}

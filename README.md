MicroRobotTracking
 ==================

A small application to track the motion of microrobots in videos

To run: 
   run 'make ModelsTest' in the ModelsTest directory 
   Then run the executable with './Modelstest' supplying a video file as the second
   argument 
            - For videos where the default channel 2 is for some
              reason unusuable, use the flag -c followed by a number from 0 to 2
              indicating which channel should be used instead (probably 1).

Using the application: 
      The first screen is used for setting the max and min areas for
      contour finding. Press any key to move to the next screen

      The second screen is for selecting which models should be
      tracked. Click inside a bounding box to select that contour, the
      contours are ordered by area, with the smallest on top. Again
      press any key to finish selection.

      The program will then compute the center and orientation data for the
      rest of the video.

      The final screen has a track-bar for scrubbing through the final
      video with bounding boxes, centers and orientation
      information drawn on the frames. Any key will exit the application.

      


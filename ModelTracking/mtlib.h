#if !defined _TOLIB
#define _TOLIB 1
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
//#include "opencv2/nonfree/nonfree.hpp"
#include <iostream>
#include <stdlib.h>
#include <string>

namespace mtlib {
  enum mask_t { NOMASK, QUAD_LL, QUAD_LR, QUAD_UL, QUAD_UR, MT_MAX = QUAD_UR };
  class Model {
  public:
    int w, h;
    double area;
    std::vector<cv::Point> centers;
    std::vector< std::vector<double>  > rotSigs;
    std::vector< std::vector<cv::Point> > contours;
    std::vector<double> rotations;
    mask_t mask;
    //returns the center at time t. If t < 0 then it returns the most recently added center
    cv::Point getCenter(int t = -1);
    //calculates the rotation of given contour and center with respect to the model
    double getContourRot(std::vector<cv::Point>, cv::Point);
    //returns the rotation at time t. If t < 0 then it returns the most recently added rotation
    double getRotation(int t = -1);
    //returns the contour at time t. If t < 0 then it returns the most recently added contour
    std::vector<cv::Point> getContour(int t = -1);
    //returns the rotation signal at time t. If t < 0 then it returns the most recently 
    //added rotation signal
    std::vector<double> getRotationSignal(int t = -1);
    //returns the area to serach for the object in the given frame
    //this is simply the bounding box of the last position enlarged by some factor
    cv::Rect getSearchArea(cv::Mat frame);
    //adds the given center, angle, rotation signal, and contour to their respective vectors
    void update(cv::Point center, double angle, std::vector<double> rotSig,
                std::vector<cv::Point> contour);
    //Returns the time index most recently added. Calling a method such as drawModel with
    //the value returned will draw the most recent update
    int curTime();
    //draws the contour of this model at some time index t on the image dst
    void drawContour(cv::Mat dst, int t);
    //draws a dot for the center and a line for the orientation at some time 
    //index t on the image dst
    void drawModel(cv::Mat dst, int t);
    //Draws a black mask over the appropriate region on the given matrix at time
    //index t. Note: does not actually draw the contour, it only 'erases' the contour
    //if it's already drawn in the right position
    void drawMask(cv::Mat, int);
    //Returns the appropriate rotated bounding box for some time index t
    cv::RotatedRect getBoundingBox(int t);
    //Draws the bounding box for some time index t on a mat frame with color c
    void drawBoundingBox(cv::Mat frame, int t, cv::Scalar c);
    //Constructs a new model given the center of the contour, its bounding box, its area
    //and the contour itself
    Model(cv::Point center, cv::RotatedRect bounding, double a, 
          std::vector<cv::Point> cont);

  private:
    const static int numTemplates = 360;
    cv::Point centerToCorner;
    const static double searchEnlargement;
    cv::RotatedRect bounding;
    std::vector<double> oSig;
    std::vector<cv::Point> contour;
  };


  //converts cv::Mat type numbers into strings
  std::string type2str(int type);


  //Opens a Video capture and loads the video stored at the
  //location given by filename into the video matrix
  //stores the fps of the video into fps, the size in s
  //and the codec in ex
  //returns true if video was sucessfully captured

  bool captureVideo(char* path, std::vector<cv::Mat>*, int* fps, cv::Size * s, int* ex);


  //Writes frames out as a sequence of images with the name frame_%0d.jpeg
  //to the given directory.
  //returns true if the video was written sucessfully
  bool writeVideo(const char* filename, std::vector<cv::Mat> frames);

  //Applies a set of filters for making tracking easier
  cv::Mat filter(cv::Mat src);

  //Applies some filters and then finds the lines in it using the HoughLine
  //opencv function
  void filterAndFindLines(cv::Mat, std::vector<cv::Vec2f> *);
  
  //compaining to filterAndFindLines. Draws the lines returned by that function
  void drawLines(cv::Mat, std::vector<cv::Vec2f> *);
  //Applies several filters to an image before finding the contours and storing the results
  //in contours and hierarchy. A channel can optionally be supplied to apply the filters to
  //a channel other than 2
  void filterAndFindContours(cv::Mat src, std::vector< std::vector<cv::Point> > * contours,
			     std::vector<cv::Vec4i> * hierarchy, 
                             cv::Point off=cv::Point(0, 0));

  //Does the same thing as above but with different filters for Elizabeth's videos
  void filterAndFindContoursElizabeth(cv::Mat src,
                                      std::vector< std::vector<cv::Point> > * contours,
                                      std::vector<cv::Vec4i> * hierarchy);

  //Draws the contours with areas in the given range onto the dst quickly
  //does not apply filters
  void drawContoursFast(cv::Mat dst, std::vector< std::vector<cv::Point> > * contours,
			     std::vector<cv::Vec4i> * hierarchy, int minArea, int maxArea);
  //Draws the contours with areas in the given range onto the dst and then
  //applies some filters
  void drawContoursAndFilter(cv::Mat dst, std::vector< std::vector<cv::Point> > * contours,
			     std::vector<cv::Vec4i> * hierarchy, int minArea, int maxArea);

  //Uses moments of the contours to find the center of the given contour
  cv::Point getCenter(std::vector<cv::Point> contour);
  
  //Convert a contour into a rotation signal that can be used to determine orientation
  std::vector<double> getRotSignal(std::vector<cv::Point>, cv::Point);

  //Takes a frame from a video (probably the first one), identifies the individual
  //contours (that lie between the min and max area) and then generates a model for each one,
  //appending it to the end of the given vector
  void generateModels(cv::Mat frame, std::vector<mtlib::Model> * models, int minArea, int maxArea);

  //Takes a frame and a list of models and updates the models with the data in the frame.
  //Locating each model depends on the current model position being relatively close to the new one
  //so this method should be applied to each frame consecutively. This method uses
  //min and max to filter out contours by area
  void updateModels(cv::Mat frame, std::vector<mtlib::Model> * models, int min, int max);

  //Displays the given frame with sliders for the minimum and maximum areas for drawing contours.
  //When the user hits a key the window closes and the function returns a Point containing the
  //minimum and maximum values.
  cv::Point getMinAndMaxAreas(cv::Mat frame);

  //Takes a filename and a vector of models and writes to the file the position and orientation of
  //the object in the follwing form: 
  //"<index> <pos_x_0> <pos_y_0> <orientation_0> <pos_x_1> <pos_y_1> <orientation_1> ..." 
  //where the index represents the index of the data in the model's position and orientation
  //vectors. Each line contains three entries (position and orientation) for each model in the
  //given vector.
  void writeFile(const char* filename, std::vector<mtlib::Model> models);

  //Takes a frame and a vector of objects found in that frame and prompts the user to select
  //Which type of masking they want for each object
  void selectMasks(cv::Mat, std::vector<mtlib::Model> *);
  //Takes a frame and a vector of objects found in that frame and prompts the user to select
  //models from that frame.
  //returns a vector containing the indicies of the selected models
  std::vector<int> selectObjects(cv::Mat frame, std::vector<mtlib::Model> * models);

  //Sets the channel upon which all filters will be applied to channel
  void setDefaultChannel(int channel);


  std::vector<cv::Point> getAffineTransformPoints(cv::Mat frame, cv::Mat (*capture)(),
                                                  int w, int h, int x, int y);

  cv::Mat fourToOne(cv::Mat);

  void getNPoints(int, std::string, std::vector<cv::Point>*, cv::Mat);

  std::vector<cv::Point> getCorners(cv::Mat frame, std::string window);
  cv::Point getGearCenter(std::vector<cv::Point>);
  double getAngleBetween(cv::Point, cv::Point);
  double getGearRotation(cv::Point, cv::Point);
  double getRelRotation(std::vector<cv::Point>, cv::Point,
                        std::vector<cv::Point>, cv::Point);
  void drawCorners(cv::Mat*, std::vector<cv::Point>);


  cv::Mat makeHistImg(std::vector<double>, int off=0);
  void showHist(const char *, std::vector<double>, int off=0);
  

  void combine(cv::Mat&, cv::Mat, cv::Mat);
}

#endif


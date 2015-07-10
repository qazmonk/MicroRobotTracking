#if !defined _TOLIB
#define _TOLIB 1
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
//#include "opencv2/nonfree/nonfree.hpp"
#include <iostream>
#include <stdlib.h>
#include <string>
#include <fstream>

namespace mtlib {
  enum mask_t { NOMASK, MQUAD_LL, MQUAD_LR, MQUAD_UL, MQUAD_UR, MHALF_L, MHALF_R,
                MHALF_U, MHALF_D, MT_MAX = MHALF_D };
  enum exposure_t { NOEXP, EQUAD_LL, EQUAD_LR, EQUAD_UL, EQUAD_UR, EHALF_L, EHALF_R,
                EHALF_U, EHALF_D, ET_MAX = EHALF_D };
  class Model {
  public:
    //Initializes some internal states for the models. This only needs to be called once
    static void init();
    //calculates the rotation of given contour and center with respect to the model
    double getContourRot(std::vector<cv::Point>, cv::Point);
    //returns the center at time t. If t < 0 then it returns the most recently added center
    cv::Point getCenter(int t = -1);
    //returns the rotation at time t. If t < 0 then it returns the most recently added rotation
    double getRotation(int t = -1);
    //returns the contour at time t. If t < 0 then it returns the most recently added contour
    std::vector<cv::Point> getContour(int t = -1);
    //returns the timestamp at time t. If t < 0 then it returns the most recently added timestamp
    unsigned long getTimestamp(int t = -1);
    //returns the found flag at time t. If t < 0 then it returns the most recently added flag
    bool getFoundFlag(int t = -1);
    //returns the rotation signal at time t. If t < 0 then it returns the most recently 
    //added rotation signal
    std::vector<double> getRotationSignal(int t = -1);
    //returns the area of the model
    double getArea();
    //cycles the mask of this contour to the next one
    void nextMask();
    //gets the current mask of the model
    mask_t getMask();
    //cycles the exposure of this contour to the next one
    void nextExposure();
    //gets the current exposure of the model
    exposure_t getExposure();
    //sets the mask
    void setMask(mask_t);
    //sets the exposure
    void setExposure(exposure_t);
    //returns the area to serach for the object in the given frame
    //this is simply the bounding box of the last position enlarged by some factor
    cv::Rect getSearchArea(cv::Mat frame);
    //adds the given center, angle, rotation signal, contour, found flag, and timestamp
    //to their respective vectors
    void update(cv::Point center, double angle, std::vector<double> rotSig,
                std::vector<cv::Point> contour, bool, unsigned long);
    //Returns the time index most recently added. Calling a method such as drawModel with
    //the value returned will draw the most recent update
    int curTime();
    //draws the contour of this model at some time index t in the given color
    //on the image dst
    void drawContour(cv::Mat dst, int t, cv::Scalar color=cv::Scalar(255, 255, 255));
    //draws the exposure of this model at some time index t on the image dst
    void drawExposure(cv::Mat dst, int t);
    //draws a dot for the center and a line for the orientation at some time 
    //index t on the image dst
    void drawModel(cv::Mat dst, int t);
    //Draws a black mask over the appropriate region on the given matrix at time
    //index t. Note: does not actually draw the contour, it only 'erases' the contour
    //if it's already drawn in the right position
    void drawMask(cv::Mat, int);
    //Returns the appropriate rotated bounding box for some time index t. If less than zero it
    //returns the most recent one
    cv::RotatedRect getBoundingBox(int t=-1);
    //Draws the bounding box for some time index t on a mat frame with color c
    void drawBoundingBox(cv::Mat frame, int t, cv::Scalar c);
    //returns an integer unique to this model
    int getId();
    //Writes out all of the collected data for this model
    void write_data(std::ofstream * strm);
    //Constructs a new model given the center of the contour, its bounding box, its area
    //the contour itself and an initial timestamp
    Model(cv::Point center, cv::RotatedRect bounding, double a, 
          std::vector<cv::Point> cont, unsigned long timestamp);
  private:
    static int count;
    const static int numTemplates = 360;
    cv::Point centerToCorner;
    const static double searchEnlargement;
    cv::RotatedRect bounding;
    std::vector<double> oSig;
    std::vector<cv::Point> contour;
    std::vector<unsigned long> timestamps;
    std::vector<bool> foundFlags;
    int w, h;
    double area;
    std::vector<cv::Point> centers;
    std::vector< std::vector<double>  > rotSigs;
    std::vector< std::vector<cv::Point> > contours;
    std::vector<double> rotations;
    mask_t mask;
    exposure_t exposure;
    int id;
  };


  //converts cv::Mat type numbers into strings
  std::string type2str(int type);


  /***********************************************************/
  /* Opens a Video capture and loads the video stored at the */
  /* location given by filename into the video matrix        */
  /* stores the fps of the video into fps, the size in s     */
  /* and the codec in ex                                     */
  /* returns true if video was sucessfully captured          */
  /***********************************************************/

  bool captureVideo(char* path, std::vector<cv::Mat>*, int* fps, cv::Size * s, int* ex);


  //Writes a .mov video to the specified filename consisting of the given frames
  //at the given fps
  bool writeVideo(const char* filename, std::vector<cv::Mat> frames, int);


  //Applies some filters and then finds the lines in it using the HoughLine
  //opencv function
  void filterAndFindLines(cv::Mat, std::vector<cv::Vec2f> *);
  
  //compaining to filterAndFindLines. Draws the lines returned by that function
  void drawLines(cv::Mat, std::vector<cv::Vec2f> *);
  //Applies just the filter for filter and find contours
  void filter(cv::Mat&, cv::Mat);
  //Applies several filters to an image before finding the contours and storing the results
  //in contours and hierarchy. A channel can optionally be supplied to apply the filters to
  //a channel other than 2
  void filterAndFindContours(cv::Mat src, std::vector< std::vector<cv::Point> > * contours,
                             cv::Point off=cv::Point(0, 0));
  //Does the same thing as above but with different filters for Elizabeth's videos
  void filterAndFindContoursElizabeth(cv::Mat src,
                                      std::vector< std::vector<cv::Point> > * contours,
                                      std::vector<cv::Vec4i> * hierarchy);

  //Draws just the contours of every model at the given time
  void drawModels(cv::Mat, std::vector<Model>, int);
  //Draws the contours with areas in the given range onto the dst quickly
  //does not apply filters
  void drawContoursFast(cv::Mat dst, std::vector< std::vector<cv::Point> > * contours,
                        int minArea, int maxArea);
  //Draws the contours with areas in the given range onto the dst and then
  //applies some filters
  void drawContoursAndFilter(cv::Mat dst, std::vector< std::vector<cv::Point> > * contours,
                             int minArea, int maxArea);

  //Uses moments of the contours to find the center of the given contour
  cv::Point getCenter(std::vector<cv::Point> contour);
  
  //Convert a contour into a rotation signal that can be used to determine orientation
  std::vector<double> getRotSignal(std::vector<cv::Point>, cv::Point);

  //Takes a frame from a video (probably the first one), identifies the individual
  //contours (that lie between the min and max area) and then generates a model for each one,
  //appending it to the end of the given vector
  void generateModels(cv::Mat frame, std::vector<mtlib::Model> * models, int minArea, 
                      int maxArea, unsigned long timestamp=0);

  //Takes a frame and a list of models and updates the models with the data in the frame.
  //Locating each model depends on the current model position being relatively close to the new one
  //so this method should be applied to each frame consecutively. This method uses
  //min and max to filter out contours by area
  void updateModels(cv::Mat frame, std::vector<mtlib::Model> * models, int min, int max,
                    unsigned long timestamp=0);

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
  
  //Determines if a given point lies inside a give rotated rect
  bool pointInRotatedRectangle(int , int , cv::RotatedRect);
  
  //Takes a set of models and performs (click) on each one when clicked and redraws the scene
  //with a call to (draw) for each model
  void selectProp(cv::Mat frame, std::vector<Model> * models, const char *,
                  void (*draw)(Model*, cv::Mat),
                  void (*click)(Model*));
  //Takes a frame and a vector of objects found in that frame and prompts the user to select
  //Which type of exposing they want for each object
  void selectExposures(cv::Mat, std::vector<mtlib::Model> *);
  //Takes a frame and a vector of objects found in that frame and prompts the user to select
  //Which type of masking they want for each object
  void selectMasks(cv::Mat, std::vector<mtlib::Model> *);
  //Takes a frame and a vector of objects found in that frame and prompts the user to select
  //models from that frame.
  //returns a vector containing the indicies of the selected models
  std::vector<int> selectObjects(cv::Mat frame, std::vector<mtlib::Model> * models);

  //Sets the channel upon which all filters will be applied to channel
  void setDefaultChannel(int channel);


  cv::Mat expandForDMD(cv::Mat, int, int);
  std::vector<cv::Point2f> getAffineTransformPoints(cv::Mat frame, int (*capture)(cv::Mat*),
                                                    std::string, int w, int h);

  std::vector<cv::Point2f> autoCalibrate(int (*)(cv::Mat *), std::string, cv::Size);
  cv::Mat fourToOne(cv::Mat);

  void getNPoints(int, std::string, std::vector<cv::Point>*, cv::Mat);

  std::vector<cv::Point> getCorners(cv::Mat frame, std::string window);
  cv::Point getGearCenter(std::vector<cv::Point>);
  double getAngleBetween(cv::Point, cv::Point);
  double getGearRotation(cv::Point, cv::Point);
  double getRelRotation(std::vector<cv::Point>, cv::Point,
                        std::vector<cv::Point>, cv::Point);
  void drawCorners(cv::Mat*, std::vector<cv::Point>, int, int);


  cv::Mat makeHistImg(std::vector<double>, int off=0);
  void showHist(const char *, std::vector<double>, int off=0);
  

  void combineHorizontal(cv::Mat&, cv::Mat, cv::Mat);
  void combineVertical(cv::Mat&, cv::Mat, cv::Mat);
  /******************
   * GEAR TRACKING
   *****************/
  
  const int SEP = 49;
  //sets all pixels in a rectangular image generated by doing a polar transformation to white
  //if they would be outside the bounds of the original image
  void setPolarEdges(cv::Mat, cv::Point);
  //Computes a slight modification of the horizontal gradient of the image
  //For each row it computes a quantity proportional to the change in value from left to right
  void rowGrad(cv::Mat, cv::Mat);
  //Sums across the rows replacing each value with either -1 if it is less than thresh and 1
  //if it's greater. It then maps these sums to a more useful range
  void rowSum(cv::Mat, cv::Mat, int);
  //Takes the locations of the peaks in polar and computes the phase shift
  int findPhase(std::vector<cv::Point2i>);
  //Finds the strong minimums of the gear outline
  std::vector<cv::Point2i> findMinimums(std::vector<cv::Point2i>);
  //Finds the strong maximums of the gear otuline
  std::vector<cv::Point2i> findMaximums(std::vector<cv::Point2i>);
  //Finds the gear outline in a polar parameterization using the given cost function
  std::vector<cv::Point2i> astar(cv::Mat const&, double (*cost)(double));
  //Converts a set of points from polar to cartesian coordinates given a center 
  //and a scaling factor for the radius
  std::vector<cv::Point2i> polarToLinear(std::vector<cv::Point2i>, cv::Point2i, int);
  //Computes the best fit circle for a set of points. The returned point has the format
  // (x, y, radius)
  cv::Point3i fitCircle(std::vector<cv::Point2i>);
  //returns true if the given file exists false otherwise
  bool file_exists(const std::string);
  
  //takes as input a file name and an extension and appends incrementing numbers to the end
  //until it finds a name that has not already been taken
  std::string safe_filename(char *, char *);
  //saves the fiven frame with the given filename and suffix adding numbers to avoid
  //ovewriting files
  void save_frame_safe(cv::Mat, const char*, const char*);

}

#endif


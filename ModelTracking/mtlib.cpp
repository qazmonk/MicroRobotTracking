#include "mtlib.h"
#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <algorithm>
#include <fstream>

using namespace cv;
using namespace std;

int DEF_CHANNEL = 2;

bool mtlib::captureVideo(char* src, vector<Mat> * dst, int* fps, Size* s, int* ex) {


  VideoCapture cap(src);

  *fps = cap.get(CV_CAP_PROP_FPS);
  *s = Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH),   
	    (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT));

  *ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));

  if (!cap.isOpened()) {
    cout << "cannot open the video file" << endl;
    return false;
  }
	
  while (1) {

    
    Mat frame;

    bool bSucess = cap.read(frame);

    if (!bSucess) {
      return true;
    }
    dst->push_back(frame);
		

		
  }
  return true;
}
bool mtlib::writeVideo(const char* name, std::vector<cv::Mat> frames) {
  for (int i = 0; i < frames.size(); i++) {

    char fileName[50];

    cout << "writing frame " << i + 1 << " of " << frames.size() << endl;
    frames[i].convertTo(frames[i], CV_8UC3);
    sprintf(fileName, "./%s/frame_%04d.jpeg", name, i);
    imwrite(fileName, frames[i]);
  }
  return true;
}

string mtlib::type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
  case CV_8U:  r = "8U"; break;
  case CV_8S:  r = "8S"; break;
  case CV_16U: r = "16U"; break;
  case CV_16S: r = "16S"; break;
  case CV_32S: r = "32S"; break;
  case CV_32F: r = "32F"; break;
  case CV_64F: r = "64F"; break;
  default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

const double mtlib::Model::searchEnlargement = 1.2;

Mat enlargeFromCenter(Mat img, Size ns) {
  Mat enlarged(ns, img.type(), Scalar(0, 0, 0));
  int offX = -(img.cols - ns.width)/2;
  int offY = -(img.rows - ns.height)/2;
  
  img.copyTo(enlarged.rowRange(offY, offY + img.rows).colRange(offX, offX + img.cols));

  return enlarged;
}

mtlib::Model::Model(Mat temp, Point init_center, RotatedRect init_bounding, double a) {
  area = a;
  bounding = init_bounding;

  templates.reserve(numTemplates);
  int c = (int)sqrt(temp.rows*temp.rows + temp.cols*temp.cols);
  Point center(c/2, c/2);
  //create an image with enough room so that the template doesn't get cut off when it's rotated
  Mat enlarged = enlargeFromCenter(temp, Size(c, c));

  //Generate all rotations of the given template
  for (int a = 0; a < numTemplates; a++) {
    Mat r = getRotationMatrix2D(center, a, 1.0);
    Mat dst(enlarged.clone());
    warpAffine(enlarged, dst, r, dst.size());
    templates.push_back(dst);
  }
  
  //initialize position and orientation vectors
  centers.push_back(init_center);
  rotations.push_back(0);
  
  //set width and height
  w = c;
  h = c;

  //vector for finding the top left corner of the object
  centerToCorner = Point(-w/2, -h/2);

}


Point mtlib::Model::getCenter() {
  return centers[centers.size()-1];
}

double mtlib::Model::getRotation() {
  return rotations[rotations.size()-1];
}

//Helper function for avoidining getting ROIs that fall outsie the frame
Point moveInside(Point c, Size s) {
  c.x = std::max(0, c.x);
  c.y = std::max(0, c.y);
 
  c.x = std::min(s.width, c.x);
  c.y = std::min(s.height, c.y);

  return c;
}


Rect mtlib::Model::getSearchArea(cv::Mat frame) {
  int nw = searchEnlargement*w, nh = searchEnlargement*h;
  Point shiftCorner(-(nw-w)/2, -(nh-h)/2);
  Point newCorner = getCenter() + centerToCorner + shiftCorner;
  Point bottomRight = newCorner + Point(nw, nh);

  newCorner = moveInside(newCorner, frame.size());
  bottomRight = moveInside(bottomRight, frame.size());

  Rect box(newCorner, bottomRight);

  return box;
  
}

void mtlib::Model::update(Point center, double rotation) {
  centers.push_back(center);
  rotations.push_back(rotation);
}

Mat mtlib::Model::getRotatedTemplate(double a) {
  //make sure a is between 0 and 360
  while (a >= 360)
    a -= 360;
  while (a < 0)
    a += 360;

  //retrieve template
  int index = a/360.0*numTemplates;
  return templates[index].clone();
  
}

RotatedRect mtlib::Model::getBoundingBox(int t) {
  double rotate = rotations[t];
  return RotatedRect(bounding.center, bounding.size, bounding.angle+rotate);
}
void mtlib::filterAndFindContours(Mat frame, vector< vector<Point> > * contours, 
				  vector<Vec4i> * hierarchy)
{
  vector<Mat> rgb;
  Mat t = Mat::zeros(frame.size(), CV_8UC1);
  split(frame, rgb);

  /*namedWindow("r", CV_WINDOW_AUTOSIZE);
  namedWindow("g", CV_WINDOW_AUTOSIZE);
  namedWindow("b", CV_WINDOW_AUTOSIZE);
  imshow("r", rgb[1]);
  imshow("g", rgb[0]);
  imshow("b", rgb[2]);
  waitKey(0);*/

  adaptiveThreshold(rgb[DEF_CHANNEL], t, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 27, 5);
  
  findContours(t, *contours, *hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
}

void mtlib::drawContoursAndFilter(Mat dst, vector< vector<Point> > * contours, 
				  vector<Vec4i> * hierarchy, int minArea, int maxArea)
{

  Mat contour_drawing = Mat::zeros(dst.size(), dst.type());
  Scalar color = Scalar(255, 255, 255);


  //loop through contours filtering out ones that are too small or too big
  for (int i = 0; i < contours->size(); i++) {
    double consize = contourArea(contours->at(i));
    if (consize >= minArea && consize <= maxArea) {
      drawContours(contour_drawing, *contours, i, color, 2, 8, *hierarchy, 0, Point());
    }
  }
  
  //apply filters to contour drawing
  Mat blurred(contour_drawing.clone());
  blur(contour_drawing, dst, Size(5, 5), Point(-1, -1));
  threshold(blurred, dst, 50, 255, THRESH_BINARY);

}
    
Point mtlib::getCenter(vector<Point> contour) {
  Moments m = moments(contour, false);
  return Point(m.m10/m.m00, m.m01/m.m00);
}

double mtlib::getRotation(Model m, Mat frame, double sweep) {
  
  //make frame bigger to help when objects go near the edges of the frame
  //originally I didn't have the factor of 4 which led to the search being less accurate
  //although I don't really understand why. If we need to make it bigger (ie objects are going 
  //slightly off screen) then I can investigate
  Mat frame_big = enlargeFromCenter(frame, Size(frame.cols+m.w/4, frame.rows+m.h/4));
  int r_cols = frame_big.cols - m.w + 1;
  int r_rows = frame_big.rows - m.h + 1;
  double minAngle = 0;
  double bestMatch = -1;
  

  double prev = m.getRotation();
  
  for (int i = 0; i < sweep; i++) {
    //generate angle inbetwwen 0 and 360
    double angle = prev - sweep/2 + i;
    while (angle >= 360)
      angle -= 360;
    while (angle < 0)
      angle += 360;
    //get template
    Mat temp = m.getRotatedTemplate(angle);
    //actually do the matching
    Mat result(r_cols, r_rows, CV_32FC1);    
    matchTemplate(frame_big, temp, result, CV_TM_SQDIFF);
    //determine best match
    double minVal, maxVal;
    Point minLoc, maxLoc;
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
    if (bestMatch < 0 || minVal < bestMatch) {

      minAngle = angle;
      bestMatch = minVal;

    }
  }

  return minAngle;
}

		  
void mtlib::Model::showModel() {
  namedWindow("Model", CV_WINDOW_AUTOSIZE);
  for (int i = 0; i < templates.size(); i+=10) {
    imshow("Model", templates[i]);
    waitKey(0);
  }
}

void mtlib::Model::drawModel(Mat dst, int t) {
  double PI = 3.14159;
  double a = rotations[t];
  Point v(std::cos(a*PI/180)*20, -std::sin(a*PI/180)*20);
  Point c = centers[t];

  line(dst, c, c + v,	 Scalar(255, 255, 255));
  circle(dst, c, 4, Scalar(255, 255, 255), -1, 8, 0);
}


void mtlib::generateModels(Mat frame, vector<Model> * models, int minArea, int maxArea) {
  vector< vector<Point> > contours;
  vector<Vec4i> hierarchy;
  //do all contour finding, drawing and filtering

  filterAndFindContours(frame, &contours, &hierarchy);
  Mat filteredContours = Mat::zeros(frame.size(), CV_8UC1);

  drawContoursAndFilter(filteredContours, &contours, &hierarchy, minArea, maxArea);
  //go through contours looking for acceptable matches

  for (int i = 0; i < contours.size(); i++) {
    double consize = contourArea(contours[i]);
    if (consize > minArea && consize < maxArea) {
      //create model and push it onto the vector
      
      Rect t = boundingRect(contours[i]);
      Point v(t.width*0.2, t.height*0.2);
      Point tl = moveInside(t.tl()-v, frame.size());
      Point br = moveInside(t.br()+v, frame.size());
      Rect bb(tl, br);
      cout << bb << endl;
      Mat temp(filteredContours, bb);
      Point c = getCenter(contours[i]);
      RotatedRect rr = minAreaRect(contours[i]);
      Model m(temp, c, rr, consize);
      models->push_back(m);
    }
  }
}


void mtlib::updateModels(Mat frame, vector<Model> * models, int minArea, int maxArea) {
  //loop through models
  Mat out = Mat::zeros(frame.size(), frame.type());
  for (int i = 0; i < models->size(); i++) {
    //Get part of image in search area
    Rect searchArea = models->at(i).getSearchArea(frame);
    Mat roi(frame.clone(), searchArea);
    //do contour finding and filtering
    vector< vector<Point> > contours;
    vector<Vec4i> hierarchy;
    filterAndFindContours(roi, &contours, &hierarchy);
    Mat roi_cont = Mat::zeros(roi.size(), CV_8UC1);
    drawContoursAndFilter(roi_cont, &contours, &hierarchy, minArea, maxArea);
 
    //serach contours for the object
    int bestCont = 0, area = contourArea(contours[i]);
    bool foundObject = false;
    for (int i = 1; i < contours.size(); i++) {
      double consize = contourArea(contours[i]);
      if (consize > minArea && consize < maxArea && consize > area) {

	foundObject = true;
	bestCont = i;
	area = consize;
      }
    }
    //if the object was found generate new center and orientation data
    //othrewise assume it hasn't moved
    Point c;
    double a;
    if (foundObject) {
      c = getCenter(contours[bestCont]) + searchArea.tl();
      a = getRotation(models->at(i), roi_cont, 30);

    } else {
      c = models->at(i).getCenter();
      a = models->at(i).getRotation();
    }
    models->at(i).update(c, a);
  }
  
}


//This is part of my hacky way of making a trackbar within this library while avoiding
//using global variables
namespace trackbarVars {
  Mat frame;
  int min = 5000;
  int max = 20000;
  int lastMin = 0;
  int lastMax = 0;
}
//callback function that updates the image with the new min and max area values
void applyFilter(int, void*) {
  int step = 1000;
  trackbarVars::min = floor(trackbarVars::min/step)*step;
  trackbarVars::max = floor(trackbarVars::max/step)*step;
  cout << "min and last " << trackbarVars::min << " " << trackbarVars::lastMin << endl;
  cout << "max and last " << trackbarVars::max << " " << trackbarVars::lastMax << endl;

  if (trackbarVars::min != trackbarVars::lastMin || trackbarVars::max != trackbarVars::lastMax) {
    cout << "min = " << trackbarVars::min << " max = " << trackbarVars::max << endl;
    vector< vector<Point> > contours; 
    vector<Vec4i> hierarchy;
    mtlib::filterAndFindContours(trackbarVars::frame, &contours, &hierarchy);
    Mat disp = Mat::zeros(trackbarVars::frame.size(), CV_8UC1);
    mtlib::drawContoursAndFilter(disp, &contours, &hierarchy, 
				 trackbarVars::min, trackbarVars::max);
    imshow("Frame", disp);
  }
  trackbarVars::lastMin = trackbarVars::min;
  trackbarVars::lastMax = trackbarVars::max;
}
//creates a window with two trackbars for selecting the min and max area values
Point mtlib::getMinAndMaxAreas(Mat frame) {

  namedWindow("Frame", CV_WINDOW_AUTOSIZE);
  createTrackbar("Min", "Frame", &trackbarVars::min, 50000, applyFilter);
  createTrackbar("Max", "Frame", &trackbarVars::max, 50000, applyFilter);
  trackbarVars::frame = frame;

  applyFilter(0, 0);
  waitKey(0);
  destroyWindow("Frame");

  return Point(trackbarVars::min, trackbarVars::max);
}

void mtlib::writeFile(const char* filename, vector<Model> models) {
  ofstream file;
  file.open(filename, ios::out);
  if (file.is_open()) {
    for (int i = 0; i < models[0].centers.size(); i++) {
      file << i;
      for (int j = 0; j < models.size(); j++) {
	file << " " << models[j].centers[i].x << " " << models[j].centers[i].y 
	     << " " << models[j].rotations[i];
      }
      file << endl;
    }
  } else {
    cout << "Error: Could not open file " << filename << endl;
  }
  file.close();
}

namespace selectROIVars {
  bool lastMouseButton = false;
  vector<mtlib::Model> * modelsToSearch;
  cv::Mat contours;
  vector<bool> selected;
}
bool pointInRotatedRectangle(int x, int y, RotatedRect rr) {
  Point2f verticies[4];
  rr.points(verticies);
  int count = 0;
  for (int i = 0; i < 4; i++) {
    int y_min = min(verticies[i].y, verticies[(i+1)%4].y);
    int y_max = max(verticies[i].y, verticies[(i+1)%4].y);
    double m = (verticies[i].y - verticies[(i+1)%4].y)/(verticies[i].x - verticies[(i+1)%4].x);
    double delX = 1/m*(y-verticies[i].y);
    double xp = delX+verticies[i].x;
    if (y > y_min && y < y_max && xp >= x) 
      count++;
  }

  return count%2 == 1;
}
void selectROICallback(int event, int x, int y, int, void*) {
  if (event != EVENT_LBUTTONDOWN) {
    selectROIVars::lastMouseButton = false;
    return;
  }

  if (selectROIVars::lastMouseButton == false) {
    cv::Mat frame = selectROIVars::contours.clone();
    vector<mtlib::Model> * models = selectROIVars::modelsToSearch;
    double min_area = -1;
    int min = -1;
    for (int i = 0; i < models->size(); i++) {
      if (pointInRotatedRectangle(x, y, models->at(i).getBoundingBox(0))
          && (min_area < 0 || models->at(i).area <= min_area)) {
        min = i;
        min_area = models->at(i).area;
      }
    }
    if (min != -1) {
      selectROIVars::selected[min] = !selectROIVars::selected[min];
    }

    for (int n = 0; n < models->size(); n++) {
      Scalar color(0, 0, 255);
      if (selectROIVars::selected[n]) {
        color = Scalar(0, 255, 0);
      }
      Point2f verticies[4];
      models->at(n).getBoundingBox(0).points(verticies);

      for (int i = 0; i < 4; i++)
        line(frame, verticies[i], verticies[(i+1)%4], color, 2);

    }
    imshow("Select ROIs", frame);
  }
  selectROIVars::lastMouseButton = true;
}
vector<int> mtlib::selectObjects(Mat frame, vector<Model> * models) {
  selectROIVars::selected.resize(models->size(), false);
  selectROIVars::modelsToSearch = models;
  namedWindow("Select ROIs", CV_WINDOW_AUTOSIZE);

  Mat dst = Mat::zeros(frame.size(), CV_8UC3);
  double minArea = models->at(0).area;
  double maxArea = models->at(0).area;
  for (int i = 1; i < models->size(); i++) {
    double t_area = models->at(i).area;
    if (t_area < minArea)
      minArea = t_area;
    if (t_area > maxArea)
      maxArea = t_area;
  }
  vector< vector<Point> > contours;
  vector< Vec4i > hierarchy;
  filterAndFindContours(frame, &contours, &hierarchy);
  cout << contours.size() << " " << minArea << " " << maxArea << endl;
  drawContoursAndFilter(dst, &contours, &hierarchy, minArea, maxArea);  
  selectROIVars::contours = dst.clone();
  for (int n = 0; n < models->size(); n++) {
    Point2f verticies[4];
    models->at(n).getBoundingBox(0).points(verticies);
    for (int i = 0; i < 4; i++)
      line(dst, verticies[i], verticies[(i+1)%4], Scalar(0, 0, 255), 2);
  }
  imshow("Select ROIs", dst);
  setMouseCallback("Select ROIs", selectROICallback, 0);
  waitKey(0);
  destroyWindow("Select ROIs");
  vector<int> selectedIndicies;
  for (int i = 0; i < selectROIVars::selected.size(); i++) {
    if (selectROIVars::selected[i]) {
      selectedIndicies.push_back(i);
    }
  }
  return selectedIndicies;
}
void mtlib::setDefaultChannel(int channel) {
  DEF_CHANNEL = channel;
}

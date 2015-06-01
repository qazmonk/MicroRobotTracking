#include "mtlib.h"
#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <algorithm>
#include <fstream>
#include <typeinfo>
#include <math.h>
#include <sys/stat.h>
#include<unistd.h>
#define PI 3.14159265

using namespace cv;
using namespace std;

int DEF_CHANNEL = 2;
int MAX_DEV = 22;
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

  char clean[50];
  sprintf(clean, "./%s/clean.sh", name);
  if (access( clean, F_OK ) != -1) {
    cout << "cleaning directory..." << endl;
    sprintf(clean, "%s %s", clean, name);
    system(clean);

  }
  cout << "writing frames" << endl;
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

const double mtlib::Model::searchEnlargement = 1.5;

Mat enlargeFromCenter(Mat img, Size ns) {
  Mat enlarged(ns, img.type(), Scalar(0, 0, 0));
  int offX = -(img.cols - ns.width)/2;
  int offY = -(img.rows - ns.height)/2;
  
  img.copyTo(enlarged.rowRange(offY, offY + img.rows).colRange(offX, offX + img.cols));

  return enlarged;
}

mtlib::Model::Model(Mat temp, Point init_center, RotatedRect init_bounding, double a,
		    vector<Point> cont) {
  area = a;
  bounding = init_bounding;
  oSig = getRotSignal(cont, init_center);
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
double angleDist(double a1, double a2) {
  double d = abs(a1-a2);
  if (d > 180) {
    d = 360-d;
  }
  return d;
}
double mtlib::Model::getContourRot(vector<Point> cont, Point c) {
  double rot = getRotation(), min_cost = -1;
  vector<double> sig = getRotSignal(cont, c), all_costs, min_costs;
  vector<int> angles, all_angles;
  int prev_off = (360 - (int)rot)%360;
  for (int o = -MAX_DEV; o <= MAX_DEV ; o++) {
    double cost = 0;
    for (int i = 0; i < 360; i++) {
      double del = sig[(o+i+prev_off+360)%360]-oSig[i];
      cost += del*del;
    }
    //cout << "ang: " << (360 - ((o+prev_off+360)%360))%360 << " cost: "  << cost << endl;
    all_costs.push_back(cost);
    all_angles.push_back((360-(o+prev_off+360)%360)%360);
    if (min_cost < 0 || cost < min_cost) min_cost = cost;
  }
  for (int i = 1; i < all_costs.size()-1; i++) {
    if (all_costs[i] - min_cost < min_cost*0.15 && all_costs[i] < all_costs[i+1]
	&& all_costs[i] < all_costs[i-1]) {
      min_costs.push_back(all_costs[i]);
      angles.push_back(all_angles[i]);
    }
  }
  double best_angle= angles[0];
  for (int i = 0; i < min_costs.size(); i++) {
    double angle = angles[i];
    //cout << angle << " " << min_costs[i] << endl;
    if (angleDist(angle, rot) < angleDist(best_angle, rot)) {
      best_angle = angle;
    }
  }
  return best_angle;
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

int mtlib::Model::curTime() {
  return centers.size() - 1;
}
void mtlib::Model::update(Point center, double rotation) {
  centers.push_back(center);
  rotations.push_back(rotation);
  cout << "Updated to " << center << " at time " << curTime() << endl;
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
  return RotatedRect(centers[t], bounding.size, bounding.angle-rotate);
}
void mtlib::Model::drawBoundingBox(Mat frame, int t, Scalar c) {
  Point2f verticies[4];
  getBoundingBox(t).points(verticies);

  for (int i = 0; i < 4; i++)
    line(frame, verticies[i], verticies[(i+1)%4], c, 2);
}
void mtlib::filterAndFindContoursElizabeth(Mat frame, vector< vector<Point> > * contours, 
				  vector<Vec4i> * hierarchy)
{
  vector<Mat> rgb;
  Mat t0 = Mat::zeros(frame.size(), CV_8UC1);
  Mat t = Mat::zeros(frame.size(), CV_8UC1);
  Mat t2 = Mat::zeros(frame.size(), CV_8UC1);  
  split(frame, rgb);

  /*namedWindow("r", CV_WINDOW_AUTOSIZE);
  namedWindow("g", CV_WINDOW_AUTOSIZE);
  namedWindow("b", CV_WINDOW_AUTOSIZE);
  imshow("r", rgb[1]);
  imshow("g", rgb[0]);
  imshow("b", rgb[2]);
  waitKey(0);*/
  
  //namedWindow("Thresh", CV_WINDOW_AUTOSIZE);
  
  //blur(rgb[DEF_CHANNEL], t0, Size(5, 5), Point(-1, -1));
  //bilateralFilter(t0, t, 12, 50, 50);
  adaptiveThreshold(rgb[DEF_CHANNEL], t2, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
		    THRESH_BINARY_INV, 91, 1);
  //imshow("Thresh", t2);
  findContours(t2, *contours, *hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
}

Mat mtlib::filter(Mat frame) {
  Mat dst;
  Canny(frame, dst, 50, 100, 3);
  return dst;
}
void mtlib::filterAndFindLines(Mat frame, vector<Vec2f> * lines) {
  Mat dst = filter(frame);
  HoughLines(dst, *lines, 1, CV_PI/180, 70, 0, 0);
}
void mtlib::drawLines(Mat dst, vector<Vec2f> * lines) {
  for( size_t i = 0; i < lines->size(); i++ )
  {
    float rho = (*lines)[i][0], theta = (*lines)[i][1];
     Point pt1, pt2;
     double a = cos(theta), b = sin(theta);
     double x0 = a*rho, y0 = b*rho;
     pt1.x = cvRound(x0 + 1000*(-b));
     pt1.y = cvRound(y0 + 1000*(a));
     pt2.x = cvRound(x0 - 1000*(-b));
     pt2.y = cvRound(y0 - 1000*(a));
     line( dst, pt1, pt2, Scalar(255,255,0), 1, CV_AA);
  }
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
  
  findContours(t, *contours, *hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
}

void mtlib::drawContoursAndFilter(Mat dst, vector< vector<Point> > * contours, 
				  vector<Vec4i> * hierarchy, int minArea, int maxArea)
{

  Mat contour_drawing = Mat::zeros(dst.size(), dst.type());
  Scalar color = Scalar(255, 255, 0);


  //loop through contours filtering out ones that are too small or too big
  for (int i = 0; i < contours->size(); i++) {
    double consize = contourArea(contours->at(i));
    if (consize >= minArea && consize <= maxArea) {
      //drawContours(contour_drawing, *contours, i, color, 2, 8, *hierarchy, 0, Point());
      vector<Point> contour = contours->at(i);
      for (int j = 0; j < contour.size(); j++) {
	circle(contour_drawing, contour[j], 1, Scalar(255, 255, 255));
      }
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
vector<double> mtlib::getRotSignal(vector<Point> contour, Point center) {
  vector<int> sig_int(360, 0);
  int scale = 100;
  for (int i = 0; i < contour.size(); i++) {
    int a = (int)(getAngleBetween(contour[i], center));
    sig_int[a]++;
  }
  int max = 0;
  for (int i = 0; i < sig_int.size(); i++) {
    if (sig_int[i] > max) max = sig_int[i];
  }
  vector<double> sig1(360, 0), sig2(360, 0);
  for (int i = 0; i < sig_int.size(); i++) {
    sig1[i] = 100.0*((double)sig_int[i])/max;
  }

  for (int i = 0; i < sig1.size(); i++) {
    sig2[i] = (sig1[(i-2+360)%360]+2*sig1[(i-1+360)%360]+3*sig1[i]+2*sig1[(i+1)%360]
	       +sig1[(i+2)%360])/9;
  }
  return sig2;
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
  double a = rotations[t];
  Point v(std::cos(a*PI/180)*20, -std::sin(a*PI/180)*20);
  Point c = centers[t];
  cout << "Drawing model at " << c << " at time " << t << endl;
  line(dst, c, c + v,	 Scalar(255, 255, 255));
  circle(dst, c, 4, Scalar(255, 255, 255), -1, 8, 0);
}


void mtlib::generateModels(Mat frame, vector<Model> * models, int minArea, int maxArea) {
  namedWindow("Histogram Ref", CV_WINDOW_AUTOSIZE);

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
      vector<double> sig = getRotSignal(contours[i], c);
      showHist("Histogram Ref", sig);
      RotatedRect rr = minAreaRect(contours[i]);
      Model m(temp, c, rr, consize, contours[i]);
      models->push_back(m);
    }
  }
}


void mtlib::updateModels(Mat frame, vector<Model> * models, int minArea, int maxArea) {
  //loop through models
  Mat out = Mat::zeros(frame.size(), frame.type());
  //namedWindow("Searching", CV_WINDOW_AUTOSIZE);
  
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
    //imshow("Searching", roi_cont);
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
    cout << "Found object: " << foundObject << endl;
    //if the object was found generate new center and orientation data
    //othrewise assume it hasn't moved
    Point c;
    double a, ap;
    namedWindow("Histogram", CV_WINDOW_AUTOSIZE);
    if (foundObject) {
      c = getCenter(contours[bestCont]) + searchArea.tl();
      vector<double> sig = getRotSignal(contours[bestCont], c - searchArea.tl());
      //showHist("Histogram", sig);
      //ap = getRotation(models->at(i), roi_cont, 45);
      a = models->at(i).getContourRot(contours[bestCont], c - searchArea.tl());
      //cout << ap << ", " << a << endl;
    } else {
      c = models->at(i).getCenter();
      a = models->at(i).getRotation();
    }
    //circle(frame, c, 4, Scalar(255, 0, 0), -1, 8, 0);
    //imshow("Found", frame);
    //waitKey(0);
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

bool new_point = false;
void on_mouse( int e, int x, int y, int d, void *ptr )
{
  if (e != EVENT_LBUTTONDOWN || new_point)
    return;
  Point*p = (Point*)ptr;
  p->x = x;
  p->y = y;
  new_point = true;
}

void mtlib::getNPoints(int n, string window, vector<Point> *dst, Mat src) {
  Point p;
  setMouseCallback(window,on_mouse, (void*)(&p));
  for (int i = 0; i < n; i++) {
    cout << "Click and press button to record a point (" << i+1 << "/" << n << ")" << endl;
    while (!new_point) { waitKey(1); }
    dst->push_back(p);
    new_point = false;
    Mat dst_mat = src.clone();
    drawCorners(&dst_mat, *dst);
    imshow(window, dst_mat);
  }
}
//This should be essentially just a copy of what you had written just cleaned up some.
//I removed all the calls to flip since the OpenCV documentation mentioned that needing
//them was an idiosyncrasy of Windows.

//The function takes a frame that is the starting capture the user wants to use
//to get the first set of points. It also takes a function pointer called capture
//that when called returns the next frame the user wants to use. In practice we will
//write a capture function that will call the firefly capture method and pass it to this method
//The dimensions and coordinates are used to position the DMD window
vector<Point> mtlib::getAffineTransformPoints(Mat frame, Mat (*capture)(),
					      int w, int h, int x, int y) {

  vector<Point> ps;
  Point p;
  ps.reserve(6);

  Mat white(608, 648, CV_8UC1, 255);
  Mat img_bw(frame.rows, frame.cols, CV_8UC1, 255);
  Mat gray_img(frame.rows, frame.cols, CV_8UC1, 255);
  
  namedWindow("ueye", CV_WINDOW_AUTOSIZE);
  namedWindow("dmd", CV_WINDOW_NORMAL);

  //Display frame
  imshow("ueye", frame);
  
  //cvMoveWindow("dmd", 1275, -32);
  //cvResizeWindow("dmd", 608, 684);
  cvMoveWindow("dmd", x, y);
  cvResizeWindow("dmd", w, h);
  
  // Collect three source points for affine transformation.
  getNPoints(3, "ueye", &ps, frame);

  //Save the image as a gray image and threshold it
  cvtColor(frame, gray_img, CV_BGR2GRAY);
  adaptiveThreshold(gray_img, img_bw, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 27, 5);

  //show an all white image on the dmd
  imshow("dmd", white);
  waitKey(300); 

  //cature another frame and display it
  frame = (*capture)();
  imshow("ueye", frame);
  waitKey(30);

  //I got rid of this since the new set up probably won't have the same quirks as the last one
  //cap>>framew;  // This line is essential to keep the video from 'feeding back'.  Timing issue?

  

  // Crop the full image to that image contained by the rectangle myROI
  //Rect myROI(90, 50, 630, 350);

  //I'm not sure why you chose these specific numbers I had to change 640 to 630 for the demo
  //with a video with width 640 since it crashes in that case.
  //They can easily be moved to variables to change them easier
  Rect myROI(1, 1, 630, 479);
  Mat img_bw_crop = img_bw(myROI).clone();

  //Display the threshholded image on DMD
  imshow("dmd", img_bw_crop);
  waitKey(1000);

  //Loop over display of camera video.  Not sure why it's necessary for the 'delay'
  //I just coppied this exactly since I really don't understand why it is necessary to
  //Capture and display three times in successsion
  for (int i = 0; i < 3; i++) {
    cout << "Display ueye image for second part of affine transformation " << i << endl;
    frame = (*capture)();

    imshow("ueye", frame);
    waitKey(100);
  }

  getNPoints(3, "ueye", &ps, frame);
  return ps;
}

vector<Point> mtlib::getCorners (cv::Mat frame, string window) {
  imshow(window, frame);
  vector<Point> ps;
  ps.reserve(12);
  getNPoints(12, window, &ps, frame);
  return ps;
}
void mtlib::drawCorners (cv::Mat* src, vector<Point> corners) {
  circle(*src, corners[0], 4, Scalar(100, 0, 0), 1);
  for (int i = 1; i < corners.size(); i++) {
    circle(*src, corners[i], 4, Scalar(0, 100, 0), 2);
  }
}
Point mtlib::getGearCenter(vector<Point> corners) {
  Point2f c(0, 0);
  double size = (double)(corners.size());
  for (int i = 0; i < (int)(size); i++) {
    c.x += 1.0/size*corners[i].x;
    c.y += 1.0/size*corners[i].y;
  }
  Point out;
  out.x = c.x;
  out.y = c.y;
  return out;
}
void printPoint2fArray(Point2f v[], int l) {
  cout << "[";
  for (int i = 0; i < l-1; i++) {
    cout << v[i] << ", ";
  }
  cout << v[l-1] << "]" << endl;

}
double mtlib::getAngleBetween(Point p1, Point p2) {
  Point v = p1-p2;
  return atan2(v.y, v.x)*180/PI + 180;
}
//deprecated use get angle between instead
double mtlib::getGearRotation(Point top, Point center) {
  Point v = top-center;
  return atan2(v.y, v.x)*180/PI;
}
double mtlib::getRelRotation(vector<Point> prev_cor, Point prev_cent,
			     vector<Point> cur_cor, Point cur_cent) {
  double angle = 0;
  for (int i = 0; i < prev_cor.size(); i++) {
    prev_cor[i] -= prev_cent;
    cur_cor[i] -= cur_cent;    
  }
  Point2f prev[3];
  Point2f cur[3];
  for (int n = 0; n < 12; n++) {
    for (int i = 0; i < 3; i++) {
      prev[i] = prev_cor[(n+4*i)%prev_cor.size()];
      cur[i] = cur_cor[(n+4*i)%cur_cor.size()];
    }
    cout << "prev test points: ";
    printPoint2fArray(prev, 3);
    cout << "cur test points: ";
    printPoint2fArray(cur, 3);

    // angle += atan2(cur_cor[n].y, cur_cor[n].x)*180/PI + 30*n;
    // cout << "angle " << n << ": " << atan2(cur_cor[n].y, cur_cor[n].x)*180/PI << endl;
    Mat transform = getAffineTransform(prev, cur);
    transform.convertTo(transform,CV_32FC1, 1, 0);

    vector<Point3f> vec;

    Point2f src(1.0, 0), dst;
    vec.push_back(Point3f(src.x, src.y, 1.0));

    Mat srcMat = Mat(vec).reshape(1).t();
    Mat dstMat = transform*srcMat;

    dst=Point2f(dstMat.at<float>(0,0),dstMat.at<float>(1,0));
    cout << "(1, 0) mapped to (" << dst.x << ", " << dst.y << ") rot = "
	 << atan2(dst.y, dst.x)*180/PI << endl;
    angle += atan2(dst.y, dst.x)*180/PI;
  }
  return angle/12;
}
Mat  mtlib::fourToOne(Mat src) {
  int rowsp = src.rows/2;
  int colsp = src.cols/2;
  Mat dst(rowsp, colsp, src.type());
  cv::resize(src, dst, dst.size(), 0, 0);
  return dst;
}

void mtlib::showHist(const char * window, vector<double> hist) {
  
  int height = 0, width = 480, max_height = 0;
  int bar_width = width/hist.size();
  for (int i = 0; i < hist.size(); i++) {
    if (hist[i] > max_height) {
      max_height  = hist[i];
    }
  }
  if (20*max_height > 640) {
    for (int i = 0; i < hist.size(); i++) {
      hist[i] = hist[i]/((double)max_height)*(640.0/20.0);
    }
    height = 640;
  } else {
    height = 20*max_height;
  }
  Mat h = Mat::zeros(Size(width, height), CV_8UC1);

  for (int i = 0; i < hist.size(); i++) {
    rectangle(h, Point(bar_width*i, height), Point(bar_width*(i+1), height-20*hist[i]), 
	      Scalar(255, 255, 255), CV_FILLED);
  }
  
  imshow(window, h);
  waitKey(0);
  
}

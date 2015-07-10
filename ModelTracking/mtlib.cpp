#include "mtlib.h"
#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <algorithm>
#include <fstream>
#include <typeinfo>
#include <math.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <climits>
#include "ttmath/ttmath.h"
#include <string>
#include <time.h>

#define PI 3.14159265

using namespace cv;
using namespace std;

int DEF_CHANNEL = 0;
int CONT_THICKNESS = 6;
int MAX_DEV = 22;
double EXPOSURE_SCALE_FACTOR = 1.5;
int mtlib::Model::count = 0;
void mtlib::Model::init() {
  count = 0;
}
bool mtlib::captureVideo(char* src, vector<Mat> * dst, int* fps, Size* s, int* ex) 
{
  VideoCapture cap(src);
  *fps = cap.get(CV_CAP_PROP_FPS);
  *s = Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH),   
            (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT));
  
  cout << cap.get(CV_CAP_PROP_FOURCC) << endl;
  *ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));
  
  if (!cap.isOpened()) 
  {
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
bool mtlib::writeVideo(const char* name, std::vector<cv::Mat> frames, int fps) {

  cv::VideoWriter output_cap(name, 
                             CV_FOURCC('m', 'p', '4', 'v'),
                             fps,
                             frames[0].size());
  if (!output_cap.isOpened()) {
    return false;
  }
  for (int i = 0; i < frames.size(); i++) {
    if (i%100 == 0) {
      cout << "Writing frame " << i << endl;
    }
    output_cap.write(frames[i]);
  }
  output_cap.release();
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
int mtlib::Model::getId() {
  return id;
}
void mtlib::Model::write_data(ofstream * strm) {
  *strm << id << ", " << (curTime()+1) << endl;;
  for (int t =0; t <= curTime(); t++) {
    Point c = getCenter(t);
    double a = getRotation(t);
    bool found = getFoundFlag(t);
    unsigned long timestamp = getTimestamp(t);
    *strm << t << ", " << c.x << ", " << c.y << ", " << a << ", " << found 
          << ", " << timestamp << endl;
  }
}
mtlib::Model::Model(Point init_center, RotatedRect init_bounding, double a,
                    vector<Point> cont, unsigned long timestamp) {
  id = count;
  count++;
  area = a;
  bounding = init_bounding;
  oSig = getRotSignal(cont, init_center);
  contour = cont;
  mask = NOMASK;
  exposure = NOEXP;
  for (int i = 0; i < contour.size(); i++) {
    contour[i] = contour[i] - init_center;
  }
  
  //initialize position and orientation vectors
  centers.push_back(init_center);
  rotations.push_back(0);
  rotSigs.push_back(oSig);
  contours.push_back(cont);
  foundFlags.push_back(false);
  timestamps.push_back(timestamp);
  Rect bounding = boundingRect(cont);
  w = bounding.width;
  h = bounding.height;
  //vector for finding the top left corner of the object
  centerToCorner = Point(-w/2, -h/2);

  // Size s(bounding.width*1.5, bounding.height*1.5);
  // Point tl = -Point(s.width/2, s.height/2);
  // Mat model = Mat::zeros(s, CV_8UC3);
  // Mat signal = Mat::zeros(s, CV_8UC3);
  // for (int i = 0; i < cont.size(); i++) {
  //   circle(model, contour[i]-tl, 3, Scalar(255, 255, 255));
  // }
  // for (int i = 0; i <  360-1; i+= 1) {
  //   float x1 = i*((s.width-1)/359.0);
  //   float x2 = (i+1)*((s.width-1)/359.0);
  //   line(signal, Point(x1, oSig[i]), Point(x2, oSig[i+1]), Scalar(255, 255, 255), 6);
  // }
  // Mat comb;
  // combineHorizontal(comb, model, signal);
  // namedWindow("Model", CV_WINDOW_AUTOSIZE);
  // imshow("Model", comb);
  // waitKey(0);
  // save_frame_safe(comb, "rotation_signal", ".png");
}

Point mtlib::Model::getCenter(int t) {
  if (t < 0) { return centers.back(); }
  return centers[t];
}
bool mtlib::Model::getFoundFlag(int t) {
  if (t < 0) { return foundFlags.back(); }
  return foundFlags[t];
}
unsigned long mtlib::Model::getTimestamp(int t) {
  if (t < 0) { return timestamps.back(); }
  return timestamps[t];
}
double angleDist(double a1, double a2) {
  double d = abs(a1-a2);
  if (d > 180) {
    d = 360-d;
  }
  return d;
}
double mtlib::Model::getContourRot(vector<Point> cont, Point c) {
  cout << cont.size() << endl;
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
  if (angles.size() <= 0) return rot;
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
double mtlib::Model::getRotation(int t) {
  if (t < 0) { return rotations.back(); }
  return rotations[t];
}
vector<Point> mtlib::Model::getContour(int t) {
  if (t < 0) { return contours.back(); }
  return contours[t];
}
vector<double> mtlib::Model::getRotationSignal(int t) {
  if (t < 0) { return rotSigs.back(); }
  return rotSigs[t];
}
double mtlib::Model::getArea() {
  return area;
}
void mtlib::Model::nextMask() {
  mask = (mtlib::mask_t)((mask + 1)%(mtlib::MT_MAX + 1));
}
mtlib::mask_t mtlib::Model::getMask() {
  return mask;
}
void mtlib::Model::setMask(mask_t m) {
  mask = m;
}
void mtlib::Model::nextExposure() {
  exposure = (mtlib::exposure_t)((exposure + 1)%(mtlib::ET_MAX + 1));
}
mtlib::exposure_t mtlib::Model::getExposure() {
  return exposure;
}
void mtlib::Model::setExposure(exposure_t e) {
  exposure = e;
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
void mtlib::Model::update(Point center, double rotation, vector<double> rotSig,
                          vector<Point> cont, bool found, unsigned long timestamp) {
  centers.push_back(center);
  rotations.push_back(rotation);
  rotSigs.push_back(rotSig);
  contours.push_back(cont);
  foundFlags.push_back(found);
  timestamps.push_back(timestamp);
  cout << "Updated " <<  id << " to " << center << ", rotation " << " at time " 
       << curTime() << endl;
}


RotatedRect mtlib::Model::getBoundingBox(int t) {
  double rotate = getRotation(t);
  return RotatedRect(getCenter(t), bounding.size, bounding.angle-rotate);
}
void mtlib::Model::drawBoundingBox(Mat frame, int t, Scalar c) {
  Point2f verticies[4];
  getBoundingBox(t).points(verticies);

  for (int i = 0; i < 4; i++)
    line(frame, verticies[i], verticies[(i+1)%4], c, 2);
}
namespace svars {
  vector<mtlib::Model> * models;
  bool lastMouseButton = false;
  Mat ref;
  void (*draw)(mtlib::Model*, Mat);
  void (*click)(mtlib::Model*);
  const char * window_name;
}
void selectCallback(int event, int x, int y, int, void*) {
  if (event != EVENT_LBUTTONDOWN) {
    svars::lastMouseButton = false;
    return;
  }

  if (svars::lastMouseButton == false) {
    cv::Mat frame = Mat::zeros(svars::ref.size(), svars::ref.type());
    vector<mtlib::Model> * models = svars::models;
    double min_area = -1;
    int min = -1;
    for (int i = 0; i < models->size(); i++) {
      if (mtlib::pointInRotatedRectangle(x, y, models->at(i).getBoundingBox(0))
          && (min_area < 0 || models->at(i).getArea() <= min_area)) {
        min_area = models->at(i).getArea();
        min = i;
      }
    }
    if (min > -1) {
      svars::click(&(models->at(min)));
      for (int i = 0; i < models->size(); i++) {
        svars::draw(&(models->at(i)), frame);
      }
      imshow(svars::window_name, frame);
    }
  }
  svars::lastMouseButton = true;
}
void mtlib::selectProp(Mat frame, vector<mtlib::Model> * models, const char * window_name,
                       void (*draw)(Model*, Mat), void (*click)(Model*)) {
  svars::models = models;
  svars::draw = draw;
  svars::click = click;
  svars::window_name = window_name;
  namedWindow(window_name, CV_WINDOW_AUTOSIZE);
  Mat dst = Mat::zeros(frame.size(), CV_8UC3);
  for (int n = 0; n < models->size(); n++) {
    draw(&(models->at(n)), dst);
  }
  svars::ref = dst.clone();
  imshow(window_name, dst);
  setMouseCallback(window_name, selectCallback, 0);
  waitKey(0);
  destroyWindow(window_name);
}
bool mtlib::pointInRotatedRectangle(int x, int y, RotatedRect rr) {
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
void select_mask_draw(mtlib::Model* m, Mat frame) {
  cout << "drawing model" << endl;
  m->drawContour(frame, 0);
  m->drawMask(frame, 0);
  m->drawBoundingBox(frame, 0, Scalar(0, 0, 255));
}
void select_mask_click(mtlib::Model* m) {
  cout << "clicked on model" << endl;
  m->nextMask();
}
void mtlib::selectMasks(Mat frame, vector<Model> * models) {
  selectProp(frame, models, "Select Masks", *select_mask_draw, *select_mask_click);
}
void select_exposure_draw(mtlib::Model* m, Mat frame) {
  m->drawContour(frame, 0, Scalar(255, 0, 0));
  m->drawExposure(frame, 0);
  m->drawBoundingBox(frame, 0, Scalar(0, 0, 255));
}
void select_exposure_click(mtlib::Model* m) {
  m->nextExposure();
}
void mtlib::selectExposures(Mat frame, vector<Model> * models) {
  selectProp(frame, models, "Select Exposires", *select_exposure_draw, *select_exposure_click);
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


void mtlib::filterAndFindLines(Mat frame, vector<Vec2f> * lines) {
  Mat dst;
  filter(dst, frame);
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
const bool DEBUG_FILTER = false;
void mtlib::filter(Mat& dst, Mat frame) {
  int lowThreshold = 75;
  int ratio = 3;
  dst.create(frame.rows, frame.cols, CV_8UC1);
  if (DEF_CHANNEL >= 0 && DEF_CHANNEL < 3) {
    if (DEBUG_FILTER) {
      namedWindow("test", CV_WINDOW_AUTOSIZE);
    }
    vector<Mat> rgb;
    split(frame, rgb);
    rgb[DEF_CHANNEL].setTo(0);
    if (DEBUG_FILTER) {
      imshow("test", frame);
      waitKey(0);
    }
    merge(rgb, frame);
    if (DEBUG_FILTER) {
      imshow("test", frame);
      waitKey(0);
    }
    Mat gray;
    cvtColor(frame, gray, CV_BGR2GRAY);
    if (DEBUG_FILTER) {
      imshow("test", gray);
      waitKey(0);
    }
    //blur(gray, gray, Size(3, 3));
    adaptiveThreshold(gray, dst, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 27, 5);
    if (DEBUG_FILTER) {
      imshow("test", dst);
      waitKey(0);
      destroyWindow("test");
    }
    //GaussianBlur(dst, dst, Size(7, 7), 0, 0);
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    erode(dst, dst, element);
    //blur(dst, dst, Size(7, 7));
    threshold(dst, dst, 40, 255, THRESH_BINARY);
  } else {
    cout << "WARNING: using an defuct def_channel" << endl;
    Mat t1, t2;
    cvtColor(frame, t1, CV_BGR2GRAY);
    Canny(t1, t1, lowThreshold, lowThreshold*ratio, 5);
    adaptiveThreshold(t1, dst, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 27, 5);
    //blur(t2, t, Size(5, 5));
  }
}
void mtlib::filterAndFindContours(Mat frame, vector< vector<Point> > * contours, 
                                  Point off)
{
  Mat t;
  filter(t, frame);
  vector<Vec4i> h;
  findContours(t, *contours, h, CV_RETR_LIST, CV_CHAIN_APPROX_NONE, off);
}
void mtlib::drawModels(Mat dst, vector<mtlib::Model> models, int t) {
  for (int i = 0; i < models.size(); i++) {
    models[i].drawContour(dst, t);
  }
}
void mtlib::drawContoursFast(Mat dst, vector< vector<Point> > * contours, 
                             int minArea, int maxArea)
{
  //loop through contours filtering out ones that are too small or too big
  for (int i = 0; i < contours->size(); i++) {
    double consize = contourArea(contours->at(i));
    if (consize >= minArea && consize <= maxArea) {
      vector<Point> contour = contours->at(i);
      for (int j = 0; j < contour.size(); j++) {
        circle(dst, contour[j], CONT_THICKNESS/2, Scalar(255, 255, 255));

      }
    }
  }
 
}
void mtlib::drawContoursAndFilter(Mat dst, vector< vector<Point> > * contours, 
                                  int minArea, int maxArea)
{
  Mat contour_drawing = Mat::zeros(dst.size(), dst.type());
  Scalar color = Scalar(255, 255, 0);
  //loop through contours filtering out ones that are too small or too big
  for (int i = 0; i < contours->size(); i++) {
    double consize = contourArea(contours->at(i));
    if (consize >= minArea && consize <= maxArea) {
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
vector<double> smooth(vector<double> data) {
  vector<double> data_out(data.size());
  for (int i = 0; i < data.size(); i++) {
    data_out[i] = (data[(i-2+360)%360]+2*data[(i-1+360)%360]+3*data[i]+2*data[(i+1)%360]
                   +data[(i+2)%360])/9;
  }
  return data_out;
}
double dist(Point p1, Point p2) {
  double dx = p1.x-p2.x, dy = p1.y-p2.y;
  return sqrt(dx*dx + dy*dy);
}
vector<double> mtlib::getRotSignal(vector<Point> contour, Point center) {
  vector<int> sig_cnt(360, 0);
  vector<double> sig_sum(360, 0.0);
  int scale = 100;
  for (int i = 0; i < contour.size(); i++) {
    int a = (int)(getAngleBetween(contour[i], center));
    double d = dist(contour[i], center);
    sig_cnt[a]++;
    sig_sum[a] += d;
  }
  /*int max = 0;
    long total = 0;
    for (int i = 0; i < sig_int.size(); i++) {
    if (sig_int[i] > max) max = sig_int[i];
    total += sig_int[i];
    }
    double avg = ((double)total)/sig_int.size();
    vector<double> sig1(360, 0), sig2(360, 0);


    for (int i = 0; i < sig_int.size(); i++) {
    sig1[i] = 50*((double)sig_int[i])/avg;
    }*/

  vector<double> sig(360, 0);
  for (int i = 0; i < sig.size(); i++) {
    if (sig_cnt[i] == 0) {
      int j = i+1;
      while (sig_cnt[j] == 0 && j < sig.size()) j++;
      double m = 0;
      double s = 0;
      if (j < sig.size()) {
        if (i > 0) {
          m = ((sig_sum[j]/sig_cnt[j]) - (sig_sum[i-1]/sig_cnt[i-1]))/(j-(i-1));
          s = sig_sum[i-1]/sig_cnt[i-1];
        } else {
          s = sig_sum[j]/sig_cnt[j];
        }
      } else {
        if (i > 0) {
          s = sig_sum[i-1]/sig_cnt[i-1];
        } else {
          cout << "All zero rotation signal" << endl;
        }
      }
      for (int n = i; n < j; n++) {
        sig[n] = s + m * (n - (i-1));
      }
      i = j-1;
    } else {
      sig[i] = sig_sum[i]/sig_cnt[i];
    }
  }
  return smooth(sig);
}

		  

void mtlib::Model::drawModel(Mat dst, int t) {
  double a = rotations[t];
  Point v(std::cos(a*PI/180)*20, -std::sin(a*PI/180)*20);
  Point c = centers[t];
  cout << "Drawing model at " << c << " at time " << t << endl;
  line(dst, c, c + v,	 Scalar(255, 255, 255));
  circle(dst, c, 4, Scalar(255, 255, 255), -1, 8, 0);
}
void mtlib::Model::drawContour(Mat dst, int t, Scalar color) {
  vector<Point> c = getContour(t);
  for (int j = 0; j < c.size(); j++) {
    circle(dst, c[j], CONT_THICKNESS/2, color, -1);
  }
}
void mtlib::Model::drawMask(Mat dst, int t) {
  Point c  = getCenter(t);
  Point2f c2f = c;
  RotatedRect rr = getBoundingBox(t);
  double a = rr.angle;
  Point2f ihat(std::cos(a*PI/180), -std::sin(-a*PI/180));
  Point2f jhat(-ihat.y, ihat.x);
  Size2f quad = Size2f(rr.size.width/2, rr.size.height/2);
  vector<Point> cont = getContour(t);
  cout << "Drawing mask " << mask << " on model " << id << endl;
  switch(mask) {
    {case MQUAD_LL:
    case MQUAD_LR:     
    case MQUAD_UL:
    case MQUAD_UR:
      int idir, jdir;
      switch(mask) {
      case MQUAD_LL: 
        idir = 1;
        jdir = 1;
        break;
      case MQUAD_LR: 
        idir = -1;
        jdir = 1;
        break;
      case MQUAD_UL: 
        idir = 1;
        jdir = -1;
        break;
      case MQUAD_UR: 
        idir = -1; 
        jdir = -1;
        break;
      default:
        cout << "Somehow reached the quad branch mistakenly" << endl;
        break;
      }
      for (int i = 0; i < cont.size(); i++) {
        Point v = cont[i] - c;
        double idp = v.x*ihat.x + v.y*ihat.y;
        double jdp = v.x*jhat.x + v.y*jhat.y;
        if (idp*idir >= 0  && jdp*jdir >= 0) {
          circle(dst, cont[i], CONT_THICKNESS/2, Scalar(0, 0, 0), -1);
        }
      }
      break;}
    {case MHALF_L:
    case MHALF_R:
    case MHALF_U:
    case MHALF_D:
      int idir, jdir;
      switch(mask) {
      case MHALF_L:
        idir = -1;
        jdir = 0;
        break;
      case MHALF_R:
        idir = 1;
        jdir = 0;
        break;
      case MHALF_U:
        idir = 0;
        jdir = 1;
        break;
      case MHALF_D:
        idir = 0;
        jdir = -1;
        break;
      default:
        cout << "Somehow reached half plane branch accidentally" << endl;
      }
      for (int i = 0; i < cont.size(); i++) {
        Point v = cont[i] - c;
        double idp = v.x*ihat.x + v.y*ihat.y;
        double jdp = v.x*jhat.x + v.y*jhat.y;
        if (idp*idir >= 0  && jdp*jdir >= 0) {
          circle(dst, cont[i], CONT_THICKNESS/2, Scalar(0, 0, 0), -1);
        }
      }      
      break;}
  case NOMASK:
    break;
  }
}
int bound(int v, int l, int u) {
  return max(min(v, u), l);
}
bool bounded(int v, int l, int u) {
  return bound(v, l , u) == v;
}
void mtlib::Model::drawExposure(Mat dst, int t) {
  Point c  = getCenter(t);
  Point2f c2f = c;
  RotatedRect rr = getBoundingBox(t);
  double a = rr.angle;
  Point2f ihat(std::cos(a*PI/180), -std::sin(-a*PI/180));
  Point2f jhat(-ihat.y, ihat.x);
  Size2f quad = Size2f(rr.size.width/2*EXPOSURE_SCALE_FACTOR, 
                       rr.size.height/2*EXPOSURE_SCALE_FACTOR);
  Size2f half_width = Size2f(rr.size.width/2*EXPOSURE_SCALE_FACTOR, 
                             rr.size.height*EXPOSURE_SCALE_FACTOR);
  Size2f half_height = Size2f(rr.size.width*EXPOSURE_SCALE_FACTOR, 
                             rr.size.height/2*EXPOSURE_SCALE_FACTOR);
  vector<Point> cont = getContour(t);
  cout << "Drawing exposure " << exposure << " on model " << id << endl;
  switch(exposure) {
    {case EQUAD_LL:
    case EQUAD_LR:     
    case EQUAD_UL:
    case EQUAD_UR:
      int idir, jdir;
      switch(exposure) {
      case EQUAD_LL: 
        idir = 1;
        jdir = 1;
        break;
      case EQUAD_LR: 
        idir = -1;
        jdir = 1;
        break;
      case EQUAD_UL: 
        idir = 1;
        jdir = -1;
        break;
      case EQUAD_UR: 
        idir = -1; 
        jdir = -1;
        break;
      default:
        cout << "Somehow reached the quad branch mistakenly" << endl;
        break;
      }
      cout << "drawing quad" << endl;
      Point2f cp = c2f+(idir*ihat*quad.width+jdir*jhat*quad.height)*0.5;
      cout << cp << endl;
      RotatedRect exposure(cp, quad, rr.angle);
      for (double i = -quad.width/2; i <= quad.width/2; i += 0.5) {
        for (double j = -quad.height/2; j <= quad.height/2; j += 0.5) {
          Point2f pxf = cp + i*ihat + j*jhat;
          Point px((int)pxf.x, (int)pxf.y);
          if (bounded(pxf.x, 0, dst.cols-1) && bounded(pxf.y, 0, dst.rows-1))
            dst.at<Vec3b>(px) = Vec3b(255, 255, 255);
        }
      }
      break;}
    {case EHALF_L:
    case EHALF_R:
    case EHALF_U:
    case EHALF_D:
      int idir, jdir;
      Size rect_size;
      switch(exposure) {
      case EHALF_L:
        idir = -1;
        jdir = 0;
        rect_size = half_width;
        break;
      case EHALF_R:
        idir = 1;
        jdir = 0;
        rect_size = half_width;
        break;
      case EHALF_U:
        idir = 0;
        jdir = 1;
        rect_size = half_height;
        break;
      case EHALF_D:
        idir = 0;
        jdir = -1;
        rect_size = half_height;
        break;
      default:
        cout << "Somehow reached half plane branch accidentally" << endl;
      }
      Point2f cp = c2f+(idir*ihat*rect_size.width+jdir*jhat*rect_size.height)*0.5;
      cout << cp << endl;
      RotatedRect exposure(cp, rect_size, rr.angle);
      for (double i = -rect_size.width/2; i <= rect_size.width/2; i += 0.5) {
        for (double j = -rect_size.height/2; j <= rect_size.height/2; j += 0.5) {
          Point2f pxf = cp + i*ihat + j*jhat;
          Point px((int)pxf.x, (int)pxf.y);
          if (bounded(pxf.x, 0, dst.cols-1) && bounded(pxf.y, 0, dst.rows-1))
            dst.at<Vec3b>(px) = Vec3b(255, 255, 255);
        }
      }
      break;}
  case NOEXP:
    break;
  }
}
bool compare_model(mtlib::Model m1, mtlib::Model m2) { return m1.getArea() > m2.getArea(); }

void mtlib::generateModels(Mat frame, vector<Model> * models, int minArea, int maxArea,
                           unsigned long timestamp) {
  vector< vector<Point> > contours;
  //do all contour finding, drawing and filtering
  filterAndFindContours(frame, &contours);
  //go through contours looking for acceptable matches
  for (int i = 0; i < contours.size(); i++) {
    double consize = contourArea(contours[i]);
    if (consize > minArea && consize < maxArea) {
      //create model and push it onto the vector
      cout << "creating model..." << flush;
      Rect t = boundingRect(contours[i]);
      Point c = getCenter(contours[i]);
      vector<double> sig = getRotSignal(contours[i], c);
      RotatedRect rr = minAreaRect(contours[i]);
      Model m(c, rr, consize, contours[i], timestamp);
      models->push_back(m);
      cout << "done" << endl;
    }
  }
  
  sort(models->begin(), models->end(), compare_model);
}

unsigned long time_milli() {
  return chrono::duration_cast<chrono::milliseconds>
    (chrono::system_clock::now().time_since_epoch()).count();
}
void mtlib::updateModels(Mat frame, vector<Model> * models, int minArea, int maxArea,
                         unsigned long timestamp) {
  //loop through models
  Mat out = Mat::zeros(frame.size(), frame.type());
  //namedWindow("Searching", CV_WINDOW_AUTOSIZE);
  unsigned long t = time_milli();

  
  for (int i = 0; i < models->size(); i++) {
    cout << "updating model " << i << endl;
    //Get part of image in search area
    Rect searchArea = models->at(i).getSearchArea(frame);
    Mat roi(frame.clone(), searchArea);
    //do contour finding and filtering
    vector< vector<Point> > contours;
    filterAndFindContours(roi, &contours, searchArea.tl());
    if (contours.size() > 0) {
      int bestCont = 0, area_diff = abs(contourArea(contours[0]) - models->at(i).getArea());
      bool foundObject = false;
      for (int n = 0; n < contours.size(); n++) {
        double consize = contourArea(contours[n]);
        double consize_diff = abs(consize - models->at(i).getArea());
        if (consize > minArea && consize < maxArea && consize_diff <= area_diff) {
	
          foundObject = true;
          bestCont = n;
          area_diff = consize_diff;
        }
      }
      double err = ((double)area_diff)/models->at(i).getArea();
      if (err > 0.3) foundObject = false;
      cout << "Found object: " << std::boolalpha << foundObject << endl;
      //if the object was found generate new center and orientation data
      //othrewise assume it hasn't moved
      Point c;
      double a, ap;
      vector<double> r;
      vector<Point> cont;
      if (foundObject) {
        cout << "err = " << err << endl;
        c = getCenter(contours[bestCont]);
        r = getRotSignal(contours[bestCont], c);
        //showHist("Histogram", sig);
        //ap = getRotation(models->at(i), roi_cont, 45);
        a = models->at(i).getContourRot(contours[bestCont], c);
        //cout << ap << ", " << a << endl;
        cont = contours[bestCont];
      } else {
        cout << "found no contours" << endl;
        c = models->at(i).getCenter();
        a = models->at(i).getRotation();
        r = models->at(i).getRotationSignal();
        cont = models->at(i).getContour();
      }
      //circle(frame, c, 4, Scalar(255, 0, 0), -1, 8, 0);
      //imshow("Found", frame);
      //waitKey(0);
      models->at(i).update(c, a, r, cont, foundObject, timestamp);
    } else {
      Point c = models->at(i).getCenter();
      double a = models->at(i).getRotation();
      vector<double> r = models->at(i).getRotationSignal();
      vector<Point> cont = models->at(i).getContour();
      models->at(i).update(c, a, r, cont, false, timestamp); 
    }
  }
  t = time_milli() - t;
  printf("%lu milliseconds to update models at time %d\n", t, 
         models->at(0).curTime());

  
}


//This is part of my hacky way of making a trackbar within this library while avoiding
//using global variables
namespace trackbarVars {
  Mat frame;
  int min = 5000;
  int max = 25000;

  
  int lastMin = 0;
  int lastMax = 0;
  const int step = 500;
  const int maxVal = 50000;
  vector < vector<Point> > contours;
  Mat disp;
  const int num_steps = maxVal/step;
  Mat cache[num_steps][num_steps];
  bool cache_filled[num_steps][num_steps];
  int min_val = min/step;
  int max_val = max/step;
}
//callback function that updates the image with the new min and max area values
void applyFilter(int, void*) {
  int step = trackbarVars::step;
  trackbarVars::min = trackbarVars::min_val*step;
  trackbarVars::max = trackbarVars::max_val*step;
  int idx1 = trackbarVars::min_val;
  int idx2 = trackbarVars::max_val;
  if (trackbarVars::cache_filled[idx1][idx2] == false) {
    trackbarVars::cache[idx1][idx2] = Mat::zeros(trackbarVars::frame.size(), CV_8UC3);
    mtlib::drawContoursFast(trackbarVars::cache[idx1][idx2], &trackbarVars::contours, 
                            trackbarVars::min, trackbarVars::max);
    trackbarVars::cache_filled[idx1][idx2] = true;
  }
  imshow("Frame", trackbarVars::cache[idx1][idx2]);      
}
bool compContour(vector<Point> c1, vector<Point> c2) {
  int s1 = contourArea(c1);
  int s2 = contourArea(c2);
  return s1 < s2;
}
//creates a window with two trackbars for selecting the min and max area values
Point mtlib::getMinAndMaxAreas(Mat frame) {

  namedWindow("Frame", CV_WINDOW_AUTOSIZE);
  trackbarVars::frame = frame;
  trackbarVars::disp = Mat::zeros(trackbarVars::frame.size(), CV_8UC1);
  mtlib::filterAndFindContours(trackbarVars::frame, &trackbarVars::contours);
  int steps = trackbarVars::num_steps;
  for (int i = 0; i < steps; i++) {
    for (int j = 0; j < steps; j++) {
      trackbarVars::cache_filled[i][j] = false;
    }
  }
  createTrackbar("Min", "Frame", &trackbarVars::min_val, trackbarVars::num_steps, applyFilter);
  createTrackbar("Max", "Frame", &trackbarVars::max_val, trackbarVars::num_steps, applyFilter);
  
  applyFilter(0, 0);
  waitKey(0);
  destroyWindow("Frame");
  cout << trackbarVars::min << ", " << trackbarVars::max << endl;
  return Point(trackbarVars::min, trackbarVars::max);
}

void mtlib::writeFile(const char* filename, vector<Model> models) {
  ofstream file;
  file.open(filename, ios::out);
  if (file.is_open()) {
    for (int i = 0; i < models[0].curTime(); i++) {
      file << i;
      for (int j = 0; j < models.size(); j++) {
        file << " " << models[j].getCenter(i).x << " " << models[j].getCenter(i).y 
             << " " << models[j].getRotation(i);
      }
      file << endl;
    }
  } else {
    cout << "Error: Could not open file " << filename << endl;
  }
  file.close();
}

namespace osVars {
  vector<int> map;
  vector<bool> selected;
}
int select_object_get_index(mtlib::Model m) {
  for (int i = 0; i < osVars::map.size(); i++) {
    if (osVars::map[i] == m.getId()) {
      return i;
    }
  }
  return -1;
}
void select_object_draw(mtlib::Model* m, Mat frame) {
  Scalar color(0, 0, 255);
  if (osVars::selected[select_object_get_index(*m)]) {
    color = Scalar(0, 255, 0);
  }
  m->drawContour(frame, 0);
  m->drawBoundingBox(frame, 0, color);
}
void select_object_click(mtlib::Model* m) {
  int idx = select_object_get_index(*m);
  osVars::selected[idx] = !osVars::selected[idx];
}
vector<int> mtlib::selectObjects(Mat frame, vector<Model> * models) {
  osVars::map.clear();
  osVars::selected.clear();
  for (int i = 0; i < models->size(); i++) {
    osVars::map.push_back(models->at(i).getId());
    osVars::selected.push_back(false);
  }
  selectProp(frame, models, "Select Models", select_object_draw, select_object_click);
  vector<int> selectedIndicies;
  for (int i = 0; i < osVars::selected.size(); i++) {
    if (osVars::selected[i]) {
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
  int start = dst->size();
  setMouseCallback(window,on_mouse, (void*)(&p));
  for (int i = 0; i < n; i++) {
    cout << "Click and press button to record a point (" << i+1 << "/" << n << ")" << endl;
    while (!new_point) { waitKey(1); }
    dst->push_back(p);
    new_point = false;
    Mat dst_mat = src.clone();
    drawCorners(&dst_mat, *dst, start, start+i+1);
    imshow(window, dst_mat);
  }
}
void waitKeyFlush() {
  while (true) {
    if (waitKey(1) == -1) {
      break;
    }
  }
}

Mat mtlib::expandForDMD(Mat frame, int w, int h) {
  Mat dst;
  dst.create(h, w, frame.type());
  dst.setTo(Scalar(255, 255, 255));
  Mat cropped = frame(Rect(0, 0, min(w, frame.cols), min(h, frame.rows)));
    Mat tmp = dst(cv::Rect(0, 0, cropped.cols, cropped.rows));
  cropped.copyTo(tmp);
  return dst;
}
//The function takes a frame that is the starting capture the user wants to use
//to get the first set of points. It also takes a function pointer called capture
//that when called returns the next frame the user wants to use. In practice we will
//write a capture function that will call the firefly capture method and pass it to this method
//The dimensions and coordinates are used to position the DMD window
vector<Point2f> mtlib::getAffineTransformPoints(Mat frame, int (*capture)(Mat*),
                                              string dmd_window, int w, int h) {

  vector<Point> ps;
  vector<Point2f> pts;
  Point p;
  ps.reserve(6);

  Mat img_bw;//(frame.rows, frame.cols, CV_8UC1, 255);
  Mat gray_img;//(frame.rows, frame.cols, CV_8UC1, 255);
  Mat white(Size(w, h), CV_8UC3, Scalar(255, 255, 255));

  namedWindow("Calibration Input", CV_WINDOW_AUTOSIZE);
  namedWindow("DMD copy", CV_WINDOW_NORMAL);
  
  //cvMoveWindow("DMD", 1275, -32);
  //cvResizeWindow("DMD", 608, 684);
  cvResizeWindow("DMD copy", w, h); 

  //Display frame
  imshow("Calibration Input", frame);
  
  // Collect three source points for affine transformation.
  getNPoints(3, "Calibration Input", &ps, frame);

  //Save the image as a gray image and threshold it
  cvtColor(frame, gray_img, CV_BGR2GRAY);
  adaptiveThreshold(gray_img, img_bw, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 27, 5);

  /*//show an all white image on the DMD
  imshow("DMD", white);
  waitKey(300); */
  //show thresholded image on DMD
  Mat inv_bw;
  bitwise_not(img_bw, inv_bw);
  Mat expanded = expandForDMD(img_bw, w, h);
  imshow("DMD copy", expanded);
  imshow(dmd_window, expanded);
  usleep(1000000);
  //cature another frame and display it
  namedWindow("Live feed", CV_WINDOW_AUTOSIZE);
  waitKeyFlush();
  while(waitKey(1000/60) == -1) {
    (*capture)(&frame);
    imshow("Live feed", frame);
  }
  waitKeyFlush();
  destroyWindow("Live feed");
  imshow("Calibration Input", frame);
  //I got rid of this since the new set up probably won't have the same quirks as the last one
  //cap>>framew;  // This line is essential to keep the video from 'feeding back'.  Timing issue?

  

  // Crop the full image to that image contained by the rectangle myROI
  //Rect myROI(90, 50, 630, 350);

  //I'm not sure why you chose these specific numbers I had to change 640 to 630 for the demo
  //with a video with width 640 since it crashes in that case.
  //They can easily be moved to variables to change them easier
  //Rect myROI(1, 1, 630, 479);
  //Mat img_bw_crop = img_bw(myROI).clone();

  //Display the threshholded image on DMD
  //imshow("DMD", img_bw_crop);
  //waitKey(1000);

  //Loop over display of camera video.  Not sure why it's necessary for the 'delay'
  //I just coppied this exactly since I really don't understand why it is necessary to
  //Capture and display three times in successsion
  /*for (int i = 0; i < 3; i++) {
    cout << "Display Calibration Input image for second part of affine transformation " << i << endl;
    //frame = (*capture)();
    imshow("Calibration Input", frame);
    waitKey(100);
  }*/

  getNPoints(3, "Calibration Input", &ps, frame);
  destroyWindow("Calibration Input");
  destroyWindow("DMD copy");
  for (int i = 0; i < ps.size(); i++) {
    pts.push_back(ps[i]);
  }
  return pts;
}
Point2f get_light_center(Mat frame, int thresh) {
  Point2f tot(0, 0);
  int cnt = 0;
  for (int r = 0; r < frame.rows; r++) {
    for (int c = 0; c < frame.cols; c++) {
      if (frame.at<uchar>(Point(c, r)) >= thresh) {
        tot = tot + Point2f(c, r);
        cnt++;
      }
    }
  }
  if (cnt == 0) {
    cout << "No light found" << endl;
    return Point2f(0.0, 0.0);
  }
  return Point2f(tot.x/cnt, tot.y/cnt);
}
vector<Point2f> mtlib::autoCalibrate(int (*capture)(Mat*), string dmd_window, Size dmd_size) {
  Point2f src_pts[] = { Point2f(dmd_size.width*0.2, dmd_size.height*0.2),
                      Point2f(dmd_size.width*0.2, dmd_size.height*0.8),
                      Point2f(dmd_size.width*0.8, dmd_size.height*0.2),
                      Point2f(dmd_size.width*0.8, dmd_size.height*0.8) };

  namedWindow("Calibration", CV_WINDOW_AUTOSIZE);
  int thresh = 255/4;
  createTrackbar("Threshold", "Calibration", &thresh, 255);
  vector<Point2f> pts;
  Point2f dst_pts[4];
  for (int i = 0; i < 4; i++) pts.push_back(src_pts[i]);
  for (int i = 0; i < 4; i++) {
    Mat dmd(dmd_size, CV_8UC3, Scalar(255, 255, 255));
    circle(dmd, src_pts[i], 10, Scalar(0, 0, 0), CV_FILLED);
    imshow(dmd_window, dmd);
    Mat cap;
    Point c;
    cout << "Press any key when the blue dot is in the center of the light area" << endl;
    while (waitKey(1000/60) == -1) {
      capture(&cap);
      Mat gray, color, thresh_img;
      cvtColor(cap, gray, CV_BGR2GRAY);

      c = get_light_center(gray, thresh);
      threshold(gray, thresh_img, thresh, 255, THRESH_BINARY);
      Mat comb;
      combineHorizontal(comb, gray, thresh_img);
      cvtColor(comb, color, CV_GRAY2BGR);
      circle(color, c, 3, Scalar(255, 0, 0));
      imshow("Calibration", color);
    }
    pts.push_back(c);
    dst_pts[i] = c;
  }
 
  Mat warp_mat = getPerspectiveTransform(dst_pts, src_pts);
  Mat checkerboard(dmd_size, CV_8UC3, Scalar(255, 255, 255));
  Size square(dmd_size.width/4, dmd_size.height/4);
  for (int i = 0; i < 4; i += 1) {
    for (int j = 0; j < 4; j += 1) {
      if ((i+j)%2 == 0) {
        Rect r(Point(i*square.width, j*square.height), square);
        rectangle(checkerboard, r, Scalar(0, 0, 0), CV_FILLED);
      }
    }
  }
  Mat warped(dmd_size, CV_8UC3);
  warpPerspective(checkerboard, warped, warp_mat, warped.size(),
                  INTER_LINEAR, BORDER_CONSTANT, Scalar(255, 255, 255));
  imshow(dmd_window, warped);
  cout << "Press space to redo the calibration, any other key to accept it" << endl;
  while (true) {
    Mat frame;
    capture(&frame);
    imshow("Calibration", frame);
    char key = waitKey(1000/30);
    if (key == ' ') {
      return autoCalibrate(capture, dmd_window, dmd_size);
    } else if (key != -1) {
      break;
    }
  }
  return pts;
}
vector<Point> mtlib::getCorners (cv::Mat frame, string window) {
  imshow(window, frame);
  vector<Point> ps;
  ps.reserve(12);
  getNPoints(12, window, &ps, frame);
  return ps;
}
void mtlib::drawCorners (cv::Mat* src, vector<Point> corners, int s, int e) {
  circle(*src, corners[s], 4, Scalar(100, 0, 0), 1);
  for (int i = s+1; i < e; i++) {
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

Mat mtlib::makeHistImg(vector<double> hist, int off) {
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
  Mat h = Mat::zeros(Size(width, height), CV_8UC3);
  for (int i = 0; i < hist.size(); i++) {
    int idx = (i+off)%hist.size();
    Point p1 = Point(bar_width*i, height), p2 = Point(bar_width*(i+1), height-20*hist[idx]);
    rectangle(h, p1, p2, Scalar(255, 255, 255), CV_FILLED);
  }
  return h;
}
void mtlib::showHist(const char * window, vector<double> hist, int off) {
  Mat h = makeHistImg(hist, off);
  imshow(window, h);
  waitKey(0);
  
}

void mtlib::combineHorizontal(cv::Mat &dst, cv::Mat img1, cv::Mat img2) {
  int rows = max(img1.rows, img2.rows);
  int cols = img1.cols+img2.cols;
  
  dst.create(rows, cols, img1.type());
  dst.setTo(Scalar(0, 0, 0));
  cv::Mat tmp = dst(cv::Rect(0, 0, img1.cols, img1.rows));
  img1.copyTo(tmp);
  tmp = dst(cv::Rect(img1.cols, 0, img2.cols, img2.rows));
  img2.copyTo(tmp);
}
void mtlib::combineVertical(cv::Mat &dst, cv::Mat img1, cv::Mat img2) {
  int rows = img1.rows + img2.rows;
  int cols = max(img1.cols,img2.cols);
  
  dst.create(rows, cols, img1.type());
  dst.setTo(Scalar(0, 0, 0));
  cv::Mat tmp = dst(cv::Rect(0, 0, img1.cols, img1.rows));
  img1.copyTo(tmp);
  tmp = dst(cv::Rect(0, img1.rows, img2.cols, img2.rows));
  img2.copyTo(tmp);
}

/************************
 GEAR TRACKING FUNCTIONS
************************/

struct Node {
  int x;
  int y;
  double cost;
  double heuristic;
  Node(int x, int y, int c, int h)
    : x(x), y(y), cost(c), heuristic(h) {};
};

struct NodeComp {
  bool operator() (const struct Node& lhs, const struct Node& rhs) const {
    return lhs.heuristic < rhs.heuristic;
  }
};


typedef ttmath::Int<TTMATH_BITS(128)> bigInt;
typedef ttmath::Big<TTMATH_BITS(64), TTMATH_BITS(128)> bigFloat;

Point3i mtlib::fitCircle(vector<Point2i> pts) {
  bigInt A=0, B=0, C=0, D=0, E=0;
  bigInt nBig = pts.size();
  int n = pts.size();
  bigInt t = 0;
  bigInt t2 = 0;

  //I know this is a somewhat absurd way of writing this.
  //There were a lot of terms in the definitions and I wanted
  //to make sure I got them right
  for (int i = 0; i < n; i++) {
    t += pts[i].x*pts[i].x;
  }
  A += nBig*t; t=0;
  for (int i = 0; i < n; i++) {
    t += pts[i].x;
  }
  A -= t*t; t=0;

  for (int i = 0; i < n; i++) {
    t += pts[i].x*pts[i].y;
  }
  B += nBig*t; t=0;
  for (int i = 0; i < n; i++) {
    t += pts[i].x;
    t2 += pts[i].y;
  }
  B -= t*t2; t=0;

  for (int i = 0; i < n; i++) {
    t += pts[i].y*pts[i].y;
  }
  C += nBig*t; t=0;
  for (int i = 0; i < n; i++) {
    t += pts[i].y;
  }
  C -= t*t; t=0; t2=0;

  for (int i = 0; i < n; i++) {
    t += (pts[i].x)*(pts[i].y)*(pts[i].y);
    t2 += (pts[i].x)*(pts[i].x)*(pts[i].x);
  }
  D += nBig*t + nBig*t2; t=0; t2=0;
  for (int i = 0; i < n; i++) {
    t += (pts[i].x);
    t2 += (pts[i].y)*(pts[i].y);
  }
  D -= t*t2; t=0; t2=0;
  for (int i = 0; i < n; i++) {
    t += (pts[i].x);
    t2 += (pts[i].x)*(pts[i].x);
  }
  D -= t*t2; t=0; t2=0;
  D = D/2;

  for (int i = 0; i < n; i++) {
    t += (pts[i].y)*(pts[i].x)*(pts[i].x);
    t2 += (pts[i].y)*(pts[i].y)*(pts[i].y);
  }
  E += nBig*t + nBig*t2; t=0; t2=0;
  for (int i = 0; i < n; i++) {
    t += (pts[i].y);
    t2 += (pts[i].x)*(pts[i].x);
  }
  E -= t*t2; t=0; t2=0;
  for (int i = 0; i < n; i++) {
    t += (pts[i].y);
    t2 += (pts[i].y)*(pts[i].y);
  }
  E -= t*t2; t=0; t2=0;
  E = E/2;

  bigInt a = (D*C-B*E)/(A*C-B*B);
  bigInt b = (A*E-B*D)/(A*C-B*B);
  int aint, bint, rint;
  a.ToInt(aint);
  b.ToInt(bint);

  bigFloat r = 0;
  for (int i = 0; i < n; i++) {
    bigFloat dx = pts[i].x - aint;
    bigFloat dy = pts[i].y - bint;
    bigFloat sum_square = dx*dx + dy*dy;
    sum_square.Sqrt();
    r = r + sum_square;
  }
  bigFloat nfloat;
  nfloat.FromInt(n);
  r = r/nfloat;

  rint = r.ToInt();


  return Point3i(aint, bint, rint);
}

vector<Point2i> mtlib::polarToLinear(vector<Point2i> pts, Point2i c, int h) {
  vector<Point2i> cart;
  for (int i = 0; i < pts.size(); i++) {
    double x = pts[i].x*cos((pts[i].y*2*PI)/h);
    double y = pts[i].x*sin((pts[i].y*2*PI)/h);
    cart.push_back(Point2i(x, y)+c);
  }
  return cart;
}
vector<Point2i> mtlib::astar(Mat const& img, double (*cost)(double)) {
  multiset<Node, NodeComp> frontier;
  unordered_set<int> visited;
  unordered_map<int,int> parents;
  const int minRad = 30;
  const int maxRad = 300;

  uchar const* p = img.row(0).data;
  for(int i = 0; i < img.cols; ++i, ++p) {
    if(*p) {
      parents.insert(pair<int,int>(i, -1));
      visited.insert(i);
      frontier.insert(Node(i, 0, ((double)*p), ((double)*p) + img.rows));
    }
  }
  
  while(frontier.size()) {
    struct Node n = *(frontier.begin());
    if(n.y == img.rows - 1) {
      vector<Point2i> path;
      path.push_back(Point2i(n.x,n.y));
      int prev = parents[n.x + n.y*img.cols];
      while(prev != -1) {
        path.push_back(Point2i(prev % img.cols, prev / img.cols));
        prev = parents[prev];
      }
      return path;
    } else {
      frontier.erase(frontier.begin());
      const int parentIndex = n.x + n.y * img.cols;

      // Add neighbors to frontier
      for(int j = n.y; j < n.y + 2; ++j) {
        const int rowOffset = j * img.cols;
        if(j < 0) {}
        else if(j >= img.rows) break;
        else {
          for(int i = n.x - 1; i < n.x + 2; ++i) {
            if(i < minRad) continue;
            else if(i >= maxRad) break;
            else if(i == n.x && j == n.y) continue;
            else {
              int pixel = i + rowOffset;
              if(visited.find(pixel) == visited.end()) {
                double pixelColor = (double)(img.at<uchar>(j,i));
                /*if(pixelColor > 200) pixelColor = 10000;
                  if(pixelColor < 40) pixelColor = 0;
                  pixelColor = pixelColor * pixelColor * pixelColor;
                  double horiz = abs(n.x-i);
                  double childCost = n.cost + pixelColor;
                  double h = childCost + img.rows-j;
                  if(pixelColor > 255/2+5) pixelColor = 1000;
                  if(pixelColor < 30) pixelColor = 1;
                  pixelColor = pixelColor * pixelColor;
                  double horiz = abs(n.x-i);*/
                double childCost = n.cost + cost(pixelColor);
                double h = childCost + img.rows-j;
                frontier.insert(Node(i, j, childCost, h));
                parents.insert(pair<int,int>(pixel, parentIndex));
                visited.insert(pixel);
              }
            }
          }
        }
      }
    }
  }
  cout << "No path found!" << endl;
  return vector<Point2i>();
}
vector<Point2i> mtlib::findMaximums(vector<Point2i> path) {
  vector<Point2i> maxs;
  float total_y = 0;
  for (int i = 0; i < path.size() - 1; i++) {
    bool strongMax = true;
    for (int j = -17; j < 17; j++) {
      if (path[i].x < path[(j+i+path.size())%path.size()].x) strongMax = false;
    }
    if (strongMax) {
      if (maxs.size() > 0) {
        total_y += (float)(maxs.back().y-path[i].y);
      }
      maxs.push_back(path[i]);

      i += 19;
    }
  }
  return maxs;
}
vector<Point2i> mtlib::findMinimums(vector<Point2i> path) {
  vector<Point2i> mins;
  float total_y = 0;
  for (int i = 0; i < path.size() - 1; i++) {
    bool strongMin = true;
    for (int j = max(0, i-20); j < min((int)(path.size()), i+20); j++) {
      if (path[i].x > path[j].x) strongMin = false;
    }
    if (strongMin) {
      if (mins.size() > 0) {
        total_y += (float)(mins.back().y-path[i].y);
      }
      mins.push_back(path[i]);

      i += 19;
    }
  }
  return mins;
}

int mtlib::findPhase(vector<Point2i> path) {
  int sep = 49;
  int min_cost = -1;
  int min_sep = 0;
  int size = path.size()-1;
  for (int i = 0; i < sep; i++) {
    int n = 0;
    int m = 0;
    int cost = 0;
    while (m < path.size()) {
      int n1_dx = (n*mtlib::SEP+i) - path[size-m].y;
      int n2_dx = ((n+1)*mtlib::SEP+i) - path[size-m].y;
      n1_dx *= n1_dx;
      n2_dx *= n2_dx;
      if (n1_dx < n2_dx) {
        cost += n1_dx;
      } else {
        n++;
        cost += n2_dx;
      }
      m++;
    }
    if (min_cost == -1 || cost < min_cost) {
      min_cost = cost;
      min_sep = i;
    }
  }
  return min_sep;
}
void mtlib::rowSum(cv::Mat src, cv::Mat dst, int thresh) {
  Mat tmp(src.rows, src.cols, CV_32SC1);
  int max = -1;
  for (int r = 0; r < dst.rows; r++) {
    for (int c = 0; c < dst.cols; c++) {
      if (src.at<uchar>(r, c) > thresh) {
        tmp.at<int>(r, c) = 1;
      } else {
        tmp.at<int>(r, c) = -1;
      }
      if (c > 0) {
        tmp.at<int>(r,c) += tmp.at<int>(r,c-1);
      }
      if (max == -1 || tmp.at<int>(r, c) > max) {
        max = tmp.at<int>(r, c);
      }
    }
  }
  for (int r = 0; r < tmp.rows; r++) {
    for (int c = 0; c < dst.cols; c++) {
      if (tmp.at<int>(r, c) < 0) {
        tmp.at<int>(r, c) = 0;
      }
      dst.at<uchar>(r, c) = 255 - (int)((((float)tmp.at<int>(r, c))/max)*255);
    }
  }
}
void mtlib::rowGrad(cv::Mat src, cv::Mat dst) {

  for (int r = 0; r < dst.rows; r++) {
    for (int c = 2; c < dst.cols-2; c++) {
      int left = (src.at<uchar>(r, c-2) + src.at<uchar>(r, c-1))/2;
      int right = (src.at<uchar>(r, c+2) + src.at<uchar>(r, c+1))/2;
      dst.at<uchar>(r, c) = 255/2 + 10*(right-left);
    }
  }
  for (int r = 0; r < dst.rows; r++) {
    dst.at<uchar>(r, 0) = 255;
    dst.at<uchar>(r, 1) = 255;
    dst.at<uchar>(r, dst.cols-2) = 255;
    dst.at<uchar>(r, dst.cols-1) = 255;
  }
}


void mtlib::setPolarEdges(cv::Mat polar, Point cent) {
  for (int r = 0; r < polar.rows; r++) {
    for (int c = 0; c < polar.cols; c++) {
      double angle = ((double)2*PI*r)/polar.rows;
      int x = c*cos(angle)+cent.x;
      int y = c*sin(angle)+cent.y;
      if (x > polar.cols-2 || y > polar.rows-2 || x < 2|| y < 2) { 
        polar.at<uchar>(r, c) = 255;
      }
    }
  }
}

inline bool mtlib::file_exists(const string name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}
string mtlib::safe_filename(char * prefix, char * suffix) {
  int count = 0;
  char buffer[100];
  sprintf(buffer, "%s-%.4d%s", prefix, count, suffix);
  while (file_exists(buffer)) {
    count++;
    sprintf(buffer, "%s-%.4d%s", prefix, count, suffix);
  }
  return buffer;
}
void mtlib::save_frame_safe(Mat frame, const char * filename, const char * suffix) {
  int count = 0;
  char buffer [100];
  sprintf(buffer, "%s-%.4d%s", filename, count, suffix);
  while (file_exists(buffer)) {
    count++;
    sprintf(buffer, "%s%.4d%s", filename, count, suffix);
  }
  cout << "Writing frame " << buffer << endl;
  imwrite(buffer, frame);
}

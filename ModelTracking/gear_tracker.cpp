#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
//#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include "mtlib.h"
#include <algorithm>
#include <time.h>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <climits>
#include "ttmath/ttmath.h"

using namespace std;
using namespace cv;
using namespace mtlib;

#define PI 3.14159265

vector<Mat> video, out, out2;
int fps, ex, pos;
Size S;

void scrub(int, void*);

Mat cartToPolar(Mat polar, Point2i center) {
  Mat dst = Mat(polar.size(), polar.type(), Scalar(0, 0, 0));
  IplImage ipsrc = polar;
  IplImage ipdst = dst;
  cvLinearPolar(&ipsrc, &ipdst, cvPoint2D32f(center.x, center.y), 
	        polar.cols, CV_INTER_CUBIC);
  return dst;
}
Mat light_on_filter(Mat frame, Point2i c) {
  Mat gray;
  cvtColor(frame, gray, CV_RGB2GRAY);
  
  Mat trans = Mat(gray.rows, gray.cols, CV_8UC1, 255);
  Mat trans2 = Mat::zeros(trans.rows, trans.cols, CV_8UC1);
  Mat trans3 = Mat::zeros(trans.rows, trans.cols, CV_8UC1);
  Mat trans4 = Mat::zeros(trans.rows, trans.cols, CV_8UC1);
  rowSum(gray, trans4, 130);
  rowGrad(trans4, trans3);

  IplImage iptrans3 = trans3, iptrans2 = trans2;
  cvLinearPolar(&iptrans3, &iptrans2, cvPoint2D32f(c.x, c.y), 
		frame.cols, CV_WARP_INVERSE_MAP + CV_INTER_CUBIC);
  vector< vector<Point> > contours; 
  vector<Vec4i> hierarchy;
  threshold(trans2, trans2, 255/2-1, 255, THRESH_BINARY);
  findContours(trans2, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
  for (int i =0; i < contours.size(); i++) {
    double consize = contourArea(contours[i]);
    if (consize >= 5000 && consize <= 250000) {
      drawContours(trans, contours, i, Scalar(0,0,0), 5, 8, hierarchy, 0, Point());
    }
  }

  IplImage iptrans = trans;
  cvLinearPolar(&iptrans, &iptrans2, cvPoint2D32f(c.x, c.y), 
	        frame.cols, CV_INTER_CUBIC);
  trans = trans2;
  setPolarEdges(trans, c);
  return trans;
}
double light_off_cost(double pixelColor) {
  if(pixelColor > 200) pixelColor = 10000;
  if(pixelColor < 40) pixelColor = 0;
  pixelColor = pixelColor * pixelColor * pixelColor;
  return pixelColor;
}
double light_on_cost(double pixelColor) {
  if(pixelColor > 255/2+5) pixelColor = 1000;
  if(pixelColor < 30) pixelColor = 1;
  pixelColor = pixelColor * pixelColor;
  return pixelColor;
}
int main(int argc, char* argv[]) {

  captureVideo(argv[1], &video, &fps, &S, &ex);

  bool write = false, partialComp = false, light = false;
  int numFrames = video.size(), startFrame = 0;
  char* output_folder;
  int light_on = -1;
  for (int i = 2; i < argc; i++) {
    if (strncmp(argv[i], "-c", 2) == 0) {
      mtlib::setDefaultChannel(stoi(argv[i+1]));
      i++;
    } else if (strncmp(argv[i], "-w", 3) == 0) {
      write = true;
      output_folder = argv[i+1];
      i++;
    } else if (strncmp(argv[i], "-n", 3) == 0) {
      partialComp = true;
      numFrames = startFrame+stoi(argv[i+1]);
      i++;
    } else if (strncmp(argv[i], "-s", 3) == 0) {
      startFrame = stoi(argv[i+1]);
      i++;
    } else if (strncmp(argv[i], "-e", 3) == 0) {
      numFrames = stoi(argv[i+1]);
      i++;
    }
    else if (strncmp(argv[i], "--light_on", 12) == 0) {
      light_on = stoi(argv[i+1]);
      i++;
    }
  }
  if (light_on == startFrame) light = true;



  namedWindow("Polar", CV_WINDOW_AUTOSIZE);
  namedWindow("Linear", CV_WINDOW_AUTOSIZE);

  Point2i c(video[0].cols/2, video[0].rows/2);  
  int lastPhase = -1;
  int top = 0;
  /************************************
   * Find center of gear in first frame
   **********************************/
  Mat fst_filtered, fst_polar, fst_cart;
  fst_cart = video[startFrame]; 
  fst_polar = cartToPolar(fst_cart, c);
  vector<Point2i> path;
  if (light) {
    fst_filtered = light_on_filter(fst_polar, c);
    path = astar(fst_filtered, light_on_cost);
  } else {
    Mat gray;
    cvtColor(fst_polar, gray, CV_RGB2GRAY);
    fst_filtered = gray;
    path = astar(fst_filtered, light_off_cost);
  }
  vector<Point2i> cart_path = polarToLinear(path, c, fst_cart.rows);
  Point3i circle = fitCircle(cart_path);
  c.x = circle.x;
  c.y = circle.y;  

  /********************************
   * PROCESS WHOLE VIDEO
   *******************************/
  for (int n = startFrame; n < numFrames; n++) {

    /*******************************************
     * PROCESS IMAGE TO PREPARE FOR EDGE FINDING
     ******************************************/
    if (light_on == n) {
      light = true;
    }
    Mat filtered, polar, cart;
    cart = video[n]; 
    polar = cartToPolar(cart, c);
    if (light) {
      filtered = light_on_filter(polar, c);
    } else {
      Mat gray;
      cvtColor(polar, gray, CV_RGB2GRAY);
      filtered = gray;
    }
    /*********************
     * FIND EDGE AND PHASE
     ********************/
    if (light) {
      path = astar(filtered, light_on_cost);
    } else {
      path = astar(filtered, light_off_cost);
    }

    vector<Point2i> maxs = findMaximums(path);
    //dst = trans_color;
    int phase = findPhase(maxs);
    //check for point wrapping around
    if (lastPhase >= 0 && abs(phase-lastPhase) > mtlib::SEP/2) {
      if (lastPhase > mtlib::SEP/2) {
	top = (top+1)%12;
      } else {
	top--;
	if (top < 0) top = 11;
      }
    }
    //convert path and maximums back to cartesian
    cart_path = polarToLinear(path, c, cart.rows);
    vector<Point2i> cart_maxs = polarToLinear(maxs, c, cart.rows);
    circle = fitCircle(cart_path);
    c.x = circle.x;
    c.y = circle.y;
    //draw path
    for (int i = 0; i < path.size(); ++i) {
      polar.at<Vec3b>(path[i]) = Vec3b(255, 0, 0);
      if (cart_path[i].x >= 0 && cart_path[i].x < cart.cols &&
	  cart_path[i].y >= 0 && cart_path[i].y < cart.rows) {
	  cart.at<Vec3b>(cart_path[i]) = Vec3b(255, 0, 0);
      }

    }
    //draw maxs
    int avg_x = 0;
    for (int i = 0; i < cart_maxs.size(); i++) {
      if (cart_maxs[i].x >= 0 && cart_maxs[i].x < cart.cols &&
	  cart_maxs[i].y >= 0 && cart_maxs[i].y < cart.rows) {
	cart.at<Vec3b>(cart_maxs[i]) = Vec3b(0, 255, 0);
      }
      polar.at<Vec3b>(maxs[i]) = Vec3b(0, 255, 0);
      avg_x += maxs[i].x;
    }
    avg_x = avg_x/maxs.size();
    //generate the best fit maximums for
    vector<Point2i> best_fit_maxs;
    for (int i = 0; i < 12; i++) {
      polar.at<Vec3b>(Point2i(circle.z, i*mtlib::SEP+phase)) = Vec3b(0, 0, 255);
      best_fit_maxs.push_back(Point2i(circle.z, i*mtlib::SEP+phase));
      if (i == top) 
	cv::circle(polar, Point(maxs[i].x, i*mtlib::SEP+phase), 5, Scalar(255, 255, 0), 2);
    }
    vector<Point2i> cart_best_fit_maxs = polarToLinear(best_fit_maxs, c, cart.rows);

    //draw best fit maxes and the key point
    for (int i = 0; i < 12; i++) {
      if (cart_best_fit_maxs[i].x >= 0 && cart_best_fit_maxs[i].x < cart.cols &&
	  cart_best_fit_maxs[i].y >= 0 && cart_best_fit_maxs[i].y < cart.rows) {
	cart.at<Vec3b>(cart_best_fit_maxs[i]) = Vec3b(0, 0, 255);
	if (i == top) 
	  cv::circle(cart, cart_best_fit_maxs[i], 5, Scalar(255, 255, 0), 2);
      }
    }
    //draw the best fit curcle
    cv::circle(cart, Point(circle.x, circle.y), circle.z, Scalar(0, 255, 0), 1);
    cv::circle(cart, Point(circle.x, circle.y), 2, Scalar(0, 255, 0), 2);
    //do output stuff
    Mat comb;
    combine(comb, polar, cart);
    out.push_back(comb);
    //cout << "done " << i << "/" << video.size() << endl;
    cout << c.x << " " << c.y << " " << getGearRotation(cart_best_fit_maxs[top], c) << endl;
    lastPhase = phase;
  }
  if (write) writeVideo(output_folder, out);
  createTrackbar("Scrubbing", "Polar", &pos, out.size()-1, scrub);
  scrub(0, 0);
  waitKey(0);
}

void scrub (int , void* ) {
  imshow("Polar", out[pos]);
}

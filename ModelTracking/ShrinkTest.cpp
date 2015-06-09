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
#define SEP 49

vector<Mat> video, out, out2;
int fps, ex, pos;
Size S;

void scrub(int, void*);

int main(int argc, char* argv[]) {

  captureVideo(argv[1], &video, &fps, &S, &ex);

  bool write = false, partialComp = false;
  int numFrames = video.size(), startFrame = 0;
  char* output_folder;
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
  }




  namedWindow("Polar", CV_WINDOW_AUTOSIZE);
  namedWindow("Linear", CV_WINDOW_AUTOSIZE);

  Point2i c(video[0].cols/2, video[0].rows/2);  
  int lastPhase = -1;
  int top = 0;
  for (int n = startFrame; n < numFrames; n++) {
    Mat shrunk = video[n];
    

    Mat dst(shrunk.size(),shrunk.type(), Scalar(0, 0, 0));
    
    IplImage ipsrc = shrunk;
    IplImage ipdst = dst;


    cvLinearPolar(&ipsrc, &ipdst, cvPoint2D32f(c.x, c.y), 
		  shrunk.cols, CV_INTER_CUBIC);

    Mat gray;
    cvtColor(dst, gray, CV_RGB2GRAY);

    Mat trans(gray.rows, gray.cols, CV_8UC1, 255);
    Mat trans2 = Mat::zeros(trans.rows, trans.cols, CV_8UC1);
    Mat trans3 = Mat::zeros(trans.rows, trans.cols, CV_8UC1);
    Mat trans4 = Mat::zeros(trans.rows, trans.cols, CV_8UC1);
    rowSum(gray, trans4, 130);
    rowGrad(trans4, trans3);

    IplImage iptrans3 = trans3, iptrans2 = trans2;
    cvLinearPolar(&iptrans3, &iptrans2, cvPoint2D32f(c.x, c.y), 
		  shrunk.cols, CV_WARP_INVERSE_MAP + CV_INTER_CUBIC);
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
		  shrunk.cols, CV_INTER_CUBIC);
    trans = trans2;
    setPolarEdges(trans, c);
    Mat trans_color;
    cvtColor(trans, trans_color, CV_GRAY2RGB);


    vector<Point2i> path = astar(trans);
    vector<Point2i> maxs = findMaximums(path);
    //dst = trans_color;
    int phase = findPhase(maxs);
    if (lastPhase >= 0 && abs(phase-lastPhase) > SEP/2) {
      if (lastPhase > SEP/2) {
	top = (top+1)%12;
      } else {
	top--;
	if (top < 0) top = 11;
      }
    }
    vector<Point2i> cart_path = polarToLinear(path, c, shrunk.rows);
    vector<Point2i> cart_maxs = polarToLinear(maxs, c, shrunk.rows);
    Point3i circle = fitCircle(cart_path);
    c.x = circle.x;
    c.y = circle.y;
    for (int i = 0; i < path.size(); ++i) {
      dst.at<Vec3b>(path[i]) = Vec3b(255, 0, 0);
      if (cart_path[i].x >= 0 && cart_path[i].x < shrunk.cols &&
	  cart_path[i].y >= 0 && cart_path[i].y < shrunk.rows) {
	  shrunk.at<Vec3b>(cart_path[i]) = Vec3b(255, 0, 0);
      }

    }
    int avg_x = 0;

    for (int i = 0; i < cart_maxs.size(); i++) {
      if (cart_maxs[i].x >= 0 && cart_maxs[i].x < shrunk.cols &&
	  cart_maxs[i].y >= 0 && cart_maxs[i].y < shrunk.rows) {
	shrunk.at<Vec3b>(cart_maxs[i]) = Vec3b(0, 255, 0);
      }
      dst.at<Vec3b>(maxs[i]) = Vec3b(0, 255, 0);
      avg_x += maxs[i].x;
    }
    avg_x = avg_x/maxs.size();
    vector<Point2i> best_fit_maxs;

    for (int i = 0; i < 12; i++) {
      dst.at<Vec3b>(Point2i(circle.z, i*SEP+phase)) = Vec3b(0, 0, 255);
      best_fit_maxs.push_back(Point2i(circle.z, i*SEP+phase));
      if (i == top) 
	cv::circle(dst, Point(maxs[i].x, i*SEP+phase), 5, Scalar(255, 255, 0), 2);
    }
    vector<Point2i> cart_best_fit_maxs = polarToLinear(best_fit_maxs, c, shrunk.rows);

    for (int i = 0; i < 12; i++) {
      if (cart_best_fit_maxs[i].x >= 0 && cart_best_fit_maxs[i].x < shrunk.cols &&
	  cart_best_fit_maxs[i].y >= 0 && cart_best_fit_maxs[i].y < shrunk.rows) {
	shrunk.at<Vec3b>(cart_best_fit_maxs[i]) = Vec3b(0, 0, 255);
	if (i == top) 
	  cv::circle(shrunk, cart_best_fit_maxs[i], 5, Scalar(255, 255, 0), 2);
      }
    }
    cv::circle(shrunk, Point(circle.x, circle.y), circle.z, Scalar(0, 255, 0), 1);
    cv::circle(shrunk, Point(circle.x, circle.y), 2, Scalar(0, 255, 0), 2);
    Mat comb;
    combine(comb, dst, shrunk);
    out.push_back(comb);
    out2.push_back(trans_color);
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
  imshow("Linear", out2[pos]);
}

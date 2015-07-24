#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include "mtlib.h"
#include <algorithm>
#include <time.h>
#include<boost/tokenizer.hpp>
#include <fstream>

#define PI 3.14159265

using namespace std;
using namespace cv;
using namespace mtlib;

vector<Mat> video;
int fps, ex, pos = 0;
Size S;
string window = "Input";

void scrub(int, void*);

void printPointVector(vector<Point> v) {
  cout << "[";
  for (int i = 0; i < v.size()-1; i++) {
    cout << v[i] << ", ";
  }
  cout << v[v.size()-1] << "]" << endl;
}
int main(int argc, char* argv[]) {
  typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
  boost::char_separator<char> sep(", []");

  string line;
  ifstream data("corner_data.txt");
  vector< vector< Point > > file_corners;
  if (data.is_open()) {
    while(getline(data,line)) {
      tokenizer tokens(line, sep);
      vector<Point> corners;
      for (tokenizer::iterator tok_iter = tokens.begin();
	   tok_iter != tokens.end(); tok_iter++) {
	Point c;
	c.x = std::stoi(*tok_iter);
	tok_iter++;
	c.y = std::stoi(*tok_iter);
	corners.push_back(c);
      }
      file_corners.push_back(corners);
    }
  }

  for (int i = 0; i < file_corners.size(); i++) {
    Point cent = getGearCenter(file_corners[i]);
    double rotation = getGearRotation(file_corners[i][0], cent);
    cout << cent.x << " " << cent.y << " " << rotation << endl;
  }
  
  
  captureVideo(argv[1], &video, &fps, &S, &ex);
  namedWindow(window, CV_WINDOW_AUTOSIZE);

  vector<Point> test_prev, test_cur;
  /*  for (int i = 0; i < 12; i++) {
    test_prev.push_back(Point(100*cos(360/12 * i * PI / 180),
			      100*sin(360/12 * i * PI / 180)));
    test_cur.push_back(Point(100*cos((360/12 * i + 10) * PI / 180),
			     100*sin((360/12 * i + 10) * PI / 180)));
  }
  cout << "prev: ";
  printPointVector(test_prev);
  cout << "cur: ";
  printPointVector(test_cur);
  double test_rot = getRelRotation(test_prev, Point(0, 0), test_cur, Point(0, 0));
  cout << "test rotation: " << test_rot << endl;*/
  
  vector<Point> prev_corners;
  vector<Point> cur_corners;
  //Point prev_cent = getGearCenter(prev_corners);
  Point cur_cent;

  vector<Point> centers;
  vector<double> rotations;
  vector<vector< Point > > corners;
  for (int i = 0; i < video.size(); i++) {
    Mat dst = video[i].clone();
    if (i > 0) drawCorners(&dst, prev_corners, 0, prev_corners.size());
    cur_corners = getCorners(dst, window);
    cur_cent = getGearCenter(cur_corners);
    double abs_rot = getGearRotation(cur_corners[0], cur_cent);
    cout << "absolute rotation " << abs_rot << endl;
    centers.push_back(cur_cent);
    rotations.push_back(abs_rot);
    prev_corners.clear();
    prev_corners.push_back(cur_corners[0]);
    corners.push_back(cur_corners);
  }
  for (int i = 0; i < corners.size(); i++) {
    printPointVector(corners[i]);
  }
  for (int i = 0; i < centers.size(); i++) {
    cout << centers[i].x << " " << centers[i].y << " " << rotations[i] << endl;
  }
}

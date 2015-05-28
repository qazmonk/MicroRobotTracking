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

typedef ttmath::Int<TTMATH_BITS(128)> bigInt;
typedef ttmath::Big<TTMATH_BITS(64), TTMATH_BITS(128)> bigFloat;
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

vector<Mat> video, out, out2;
int fps, ex, pos;
Size S;

void scrub(int, void*);

Point3i fitCircle(vector<Point2i> pts) {
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

vector<Point2i> polarToLinear(vector<Point2i> pts, Point2i c, int h) {
  vector<Point2i> cart;
  for (int i = 0; i < pts.size(); i++) {
    double x = pts[i].x*cos((pts[i].y*2*PI)/h);
    double y = pts[i].x*sin((pts[i].y*2*PI)/h);
    cart.push_back(Point2i(x, y)+c);
  }
  return cart;
}
vector<Point2i> astar(Mat const& img) {
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
                // if(pixelColor > 250) pixelColor = 10000;
		// if(pixelColor < 30) pixelColor = 2;
                // pixelColor = pixelColor * pixelColor * pixelColor;
		// double horiz = abs(n.x-i);
                // double childCost = n.cost + pixelColor;
		// double h = childCost + img.rows-j;
                if(pixelColor > 255/2+5) pixelColor = 1000;
		if(pixelColor < 30) pixelColor = 1;
                pixelColor = pixelColor * pixelColor;
		double horiz = abs(n.x-i);
                double childCost = n.cost + pixelColor;
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
vector<Point2i> findMaximums(vector<Point2i> path) {
  vector<Point2i> maxs;
  float total_y = 0;
  for (int i = 1; i < path.size() - 1; i++) {
    bool strongMax = true;
    for (int j = max(0, i-20); j < std::min((int)(path.size()), i+20); j++) {
      if (path[i].x < path[j].x) strongMax = false;
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
vector<Point2i> findMinimums(vector<Point2i> path) {
  vector<Point2i> mins;
  float total_y = 0;
  for (int i = 1; i < path.size() - 1; i++) {
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

int findPhase(vector<Point2i> path) {
  int sep = 49;
  int min_cost = -1;
  int min_sep = 0;
  int size = path.size()-1;
  for (int i = 0; i < sep; i++) {
    int n = 0;
    int m = 0;
    int cost = 0;
    while (m < path.size()) {
      int n1_dx = (n*SEP+i) - path[size-m].y;
      int n2_dx = ((n+1)*SEP+i) - path[size-m].y;
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
void rowAvgDiff(cv::Mat src, cv::Mat dst) {
  Mat dst2 = dst.clone();
  for (int r = 0; r < dst2.rows; r++) {
    for (int c = 0; c < dst2.cols; c++) {
      int l_avg = 0, t = 0;
      for (int i = max(0, c-20); i < c; i++) {
	l_avg += src.at<uchar>(r, i);
	t++;
      }
      t = max(t, 1);
      l_avg = l_avg/t;
      int r_avg = 0;
      t = 0;
      for (int i = c; i < min(dst2.cols, c+20); i++) {
	r_avg += src.at<uchar>(r, i);
	t++;
      }
      t = max(t, 1);
      r_avg = r_avg/t;

      int tmp = 8*(r_avg-l_avg)+255/2;
      tmp = max(0, tmp);
      tmp = min(255, tmp);
      if (r_avg == 0 || l_avg == 0) {
	tmp = 255;
      }
      dst2.at<uchar>(r, c) = tmp;
      
    }
  }

  blur(dst2, dst, Size(15, 15), Point(-1,-1));

}
void rowSum(cv::Mat src, cv::Mat dst, int thresh) {
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
void rowGrad(cv::Mat src, cv::Mat dst) {

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

void combine(cv::Mat &dst, cv::Mat img1, cv::Mat img2) {
  int rows = img1.rows;
  int cols = img2.cols;
  dst.create(rows, cols*2, img1.type());
  cv::Mat tmp = dst(cv::Rect(0, 0, cols, rows));
  img1.copyTo(tmp);
  tmp = dst(cv::Rect(cols, 0, cols, rows));
  img2.copyTo(tmp);
}
void setPolarEdges(cv::Mat polar, Point cent) {
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

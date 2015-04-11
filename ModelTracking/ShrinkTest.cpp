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

vector<Point2i> polarTolinear(vector<Point2i> pts, Point2i c, int h) {
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
      for(int j = n.y - 1; j < n.y + 2; ++j) {
        const int rowOffset = j * img.cols;
        if(j < 0) continue;
        else if(j >= img.rows) break;
        else {
          for(int i = n.x - 1; i < n.x + 2; ++i) {
            if(i < minRad) continue;
            else if(i >= maxRad) break;
            else if(i == n.x && j == n.y) continue;
            else if(img.at<uchar>(j,i) == 0) continue;
            else {
              int pixel = i + rowOffset;
              if(visited.find(pixel) == visited.end()) {
                double pixelColor = (double)(img.at<uchar>(j,i));
                if(pixelColor > 200) pixelColor = 10000;
                pixelColor = pixelColor * pixelColor * pixelColor;
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


int main(int argc, char* argv[]) {

  captureVideo(argv[1], &video, &fps, &S, &ex);

  bool write = false, partialComp = false;
  int numFrames = video.size();
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
      numFrames = stoi(argv[i+1]);
      i++;
    }
  }




  namedWindow("Polar", CV_WINDOW_AUTOSIZE);
  namedWindow("Linear", CV_WINDOW_AUTOSIZE);

  Point2i c(video[0].cols/2, video[0].rows/2);  
  for (int i = 0; i < numFrames; i++) {
    Mat shrunk = video[i];
    

    Mat dst(shrunk.size(),shrunk.type(), Scalar(0, 0, 0));
    
    IplImage ipsrc = shrunk;
    IplImage ipdst = dst;


    cvLinearPolar(&ipsrc, &ipdst, cvPoint2D32f(c.x, c.y), 
		  shrunk.cols, CV_INTER_CUBIC);

    Mat gray;
    cvtColor(dst, gray, CV_RGB2GRAY);
    vector<Point2i> path = astar(gray);

    vector<Point2i> cart_path = polarTolinear(path, c, shrunk.rows);
    Point3i circle = fitCircle(cart_path);
    cout << "fit: " << circle << endl;
    c.x = circle.x;
    c.y = circle.y;
    for (int i = 0; i < path.size(); ++i) {
      dst.at<Vec3b>(path[i]) = Vec3b(255, 0, 0);
      if (cart_path[i].x >= 0 && cart_path[i].x < shrunk.cols &&
	  cart_path[i].y >= 0 && cart_path[i].y < shrunk.rows) {
	shrunk.at<Vec3b>(cart_path[i]) = Vec3b(255, 0, 0);
      }
    }
    cv::circle(shrunk, Point(circle.x, circle.y), circle.z, Scalar(0, 255, 0), 1);
    cv::circle(shrunk, Point(circle.x, circle.y), 2, Scalar(0, 255, 0), 2);
    out.push_back(dst);
    out2.push_back(shrunk);
    cout << "done " << i << "/" << video.size() << endl;
  }
  if (write) writeVideo(output_folder, out2);
  createTrackbar("Scrubbing", "Polar", &pos, numFrames-1, scrub);
  scrub(0, 0);
  waitKey(0);
}

void scrub (int , void* ) {
  imshow("Polar", out[pos]);
  imshow("Linear", out2[pos]);
}


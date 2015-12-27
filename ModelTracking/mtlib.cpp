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
#include <queue>
#include <set>
#define PI 3.14159265

using namespace cv;
using namespace std;

int DEF_CHANNEL = 0;
int CONT_THICKNESS = 4;
int MAX_DEV = 22;
double EXPOSURE_SCALE_FACTOR = 7.5;
int mtlib::Model::count = 0;
string output_figure_filename = "tmp";


void mtlib::Model::init() {
  count = 0;
}
bool mtlib::captureVideo(char* src, vector<Mat> * dst, int* fps, Size* s, int* ex, 
                         int num_frames) {
  VideoCapture cap(src);
  *fps = cap.get(CV_CAP_PROP_FPS);
  *s = Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH),   
            (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT));
  *ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));
  
  if (!cap.isOpened()) 
  {
    cout << "cannot open the video file" << endl;
    return false;
  }
  if (num_frames == -1) {
    while (1) {
      Mat frame;
      bool bSucess = cap.read(frame);
      if (!bSucess) {
        return true;
      }
      dst->push_back(frame);
    }
  } else {
    for (int i = 0; i < num_frames; i++) {
      Mat frame;
      bool bSucess = cap.read(frame);
      if (!bSucess) {
        return true;
      }
      dst->push_back(frame);
    }
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
string mtlib::Model::get_info_string(int t) {
  char buff[100];
  sprintf(buff, "ID: %d POS: [%d, %d] ROT: %f TIME: %lu FOUND: %s COST: %f",
          id, getCenter(t).x, getCenter(t).y, getRotation(t), getTimestamp(t),
          getFoundFlag(t) ? "true" : "false", getCost(t));
  string out = buff;
  return out;
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
  foundFlags.push_back(true);
  timestamps.push_back(timestamp);
  costs.push_back(0);
  Rect bounding = boundingRect(cont);
  w = sqrt(bounding.width*bounding.width + bounding.height*bounding.height);
  h = w;
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
double  mtlib::Model::getCost(int t) {
  if (t < 0) { return costs.back(); }
  return costs[t];
}
double angleDist(double a1, double a2) {
  double d = abs(a1-a2);
  if (d > 180) {
    d = 360-d;
  }
  return d;
}
Point2d mtlib::Model::getContourRot(vector<Point> cont, Point c) {
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
  if (angles.size() <= 0) return Point2d(rot, -1);
  double best_angle= angles[0], best_cost = min_costs[0];
  for (int i = 0; i < min_costs.size(); i++) {
    double angle = angles[i];
    cout << angle << " " << min_costs[i] << endl;
    if (angleDist(angle, rot) < angleDist(best_angle, rot)) {
      best_angle = angle;
      best_cost = min_costs[i];
    }
  }
  Point2d pt(best_angle, sqrt(best_cost));
  return pt;
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

  cout << newCorner << " " << bottomRight << " " << Size(nw, nh) << endl;
  return box;
}

int mtlib::Model::curTime() {
  return centers.size() - 1;
}
void mtlib::Model::update(Point center, double rotation, vector<double> rotSig,
                          vector<Point> cont, bool found, unsigned long timestamp,
                          double cost) {
  centers.push_back(center);
  rotations.push_back(rotation);
  rotSigs.push_back(rotSig);
  contours.push_back(cont);
  foundFlags.push_back(found);
  timestamps.push_back(timestamp);
  costs.push_back(cost);
}


RotatedRect mtlib::Model::getBoundingBox(int t) {
  double rotate = getRotation(t);
  return RotatedRect(getCenter(t), bounding.size, bounding.angle-rotate);
}
void mtlib::Model::drawBoundingBox(Mat frame, Scalar c, int t) {
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
  m->drawBoundingBox(frame, Scalar(0, 0, 255), 0);
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
  m->drawBoundingBox(frame, Scalar(0, 0, 255), 0);
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
    GaussianBlur(gray, gray, Size(9, 9), 0);
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
struct hole_node {
  int x;
  int y;
  int r;
  int c;
  int width;
  int d;
  int id;
  int g, h, f;
  vector<hole_node*> children;
  vector<hole_node*> neighbors;
  int size;
  hole_node(int x, int y, int r, int c,
            int step, int d)
    : x(x), y(y), r(r), c(c), d(d) {
    id = c + r*step;
  };
  bool operator==(const hole_node &o) const
  {
    return (o.x == x && o.y == y);
  }
};
struct hole_node_edge {
  hole_node *u, *v;
  int x1, x2, y1, y2;
  hole_node_edge(hole_node *u, hole_node *v) 
    : u(u), v(v) {
    x1 = u->x;
    y1 = u->y;
    x2 = v->x;
    y2 = v->y;
  };
  bool operator==(const hole_node_edge &o) const {
    return (x1 == o.x1 && x2 == o.x2 && y1 == o.y1 && y2 == o.y2);
  }
};
struct hole_node_edge_hash {
  std::size_t operator() (hole_node_edge key) const {
    size_t max_hash = -1;
    size_t return_vaule = (size_t)(((key.u->id
                                     ^ (key.v->id << 1)) >> 1)%(max_hash));
    return return_vaule;

  }
};
struct hole_node_hash {
  std::size_t operator() (hole_node* key) const {
    size_t max_hash = -1;
    return (size_t)((key->id)%(max_hash));
  }
};
struct hole_node_eq {
  bool operator() (hole_node* n1, hole_node* n2) const {
    return n1->id == n2->id;
  }
};
struct hole_node_astar_cmp {
  bool operator()(hole_node* n1, hole_node* n2) {
    return n1->f > n2->f;
  }
};
bool cmp_hole_node(hole_node* n1, hole_node* n2) {
  return n1->id < n2->id;
}
typedef unordered_map<hole_node_edge, int, hole_node_edge_hash> edge_hashmap;
typedef unordered_map<hole_node*, hole_node*, hole_node_hash, hole_node_eq> vertex_hashmap;
typedef pair<int, int> pixel;
typedef hole_node*** graph;

void update_sizes(hole_node* root) {
  int sum = 0;
  for (int i = 0; i < root->children.size(); i++) {
    update_sizes(root->children[i]);
    sum += root->children[i]->size;
  }
  root->size = sum + root->children.size();
}
int exterior_distance(int r, int c, int rows, int cols) {
  return min(r, min(c, min(rows-1-r, cols-1-c)));
}
bool residual_path(hole_node* s, hole_node* t, int rows, int cols,
                   edge_hashmap * c, edge_hashmap * f, vertex_hashmap *parents) {
  priority_queue<hole_node*, vector<hole_node*>, hole_node_astar_cmp> frontier;
  bool visited[rows][cols], in_frontier[rows][cols];
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      visited[r][c] = false;
      in_frontier[r][c] = false;
    }
  }
  s->g = 0;
  int min_dist = -1;
  for (int i = 0; i < s->neighbors.size(); i++) {
    int d = exterior_distance(s->r, s->c, rows, cols);
    if (min_dist < 0 || d < min_dist) {
      min_dist = d;
    }
  }
  s->h = min_dist + 2;
  s->f = s->h;
  frontier.push(s);
  bool t_in_frontier = false;
  while(frontier.size()) {
    hole_node * n_ptr = frontier.top();
    hole_node n = *n_ptr;
    if (n_ptr == t) {
      return true;
    } else {
      frontier.pop();
      if (n.id > 0) {
        visited[n.r][n.c] = true;
        in_frontier[n.r][n.c] = false;
      }
      for (int i = 0; i < n.neighbors.size(); i++) {
        hole_node * u = n.neighbors[i];
        if (u != t && (u == s || visited[u->r][u->c])) {
          continue;
        }
        int flow = (f->find(hole_node_edge(n_ptr, u)))->second;
        int cap = (c->find(hole_node_edge(n_ptr, u)))->second;
        int cf = cap - flow;
        if (cf > 0) {
          int t_g = n.g + 1;
          if (u == t && (!t_in_frontier || t_g < t->g)) {
            (*parents)[t] = n_ptr;
            t->g = t_g;
            t->f = t_g + 1;
            if (!t_in_frontier) {
              frontier.push(t);
              t_in_frontier = true;
            }
          } else if (u == s) {
          } else if (!in_frontier[u->r][u->c] || t_g < u->g) {
            (*parents)[u] = n_ptr;
            u->g = t_g;
            u->f = t_g + exterior_distance(u->r, u->c, rows, cols) + 1;
            if (!in_frontier[u->r][u->c]) {
              frontier.push(u);
              in_frontier[u->r][u->c] = true;
            }
          }
        }
      }
    }
  }
  return false;
}
vector<hole_node*> residual_reachable(hole_node* s, edge_hashmap * c,
                                      edge_hashmap * f, int rows, int cols) {
  queue<hole_node*> q;
  bool discovered[rows][cols];
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      discovered[r][c] = false;
    }
  }
  q.push(s);
  discovered[s->r][s->c] = true;
  vector<hole_node*> partition;
  while (q.size() > 0) {
    hole_node * u = q.front();
    q.pop();
    for (int i = 0; i < u->neighbors.size(); i++) {
      int flow = (f->find(hole_node_edge(u, u->neighbors[i])))->second;
      int cap = (c->find(hole_node_edge(u, u->neighbors[i])))->second;
      int cf = cap - flow;
      if (cf > 0 && discovered[u->neighbors[i]->r][u->neighbors[i]->c] == false) {
        q.push(u->neighbors[i]);
        discovered[u->neighbors[i]->r][u->neighbors[i]->c] = true;;
      }
    }
    partition.push_back(u);
  }
  return partition;
}
vector<hole_node*> min_cut(vector<hole_node*> srcs, vector<hole_node*> tgts, 
                           hole_node** graph, int rows, int cols) {
  edge_hashmap f, cap;
  vertex_hashmap p;
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      hole_node* cur = graph[r*cols + c];
      if (cur != NULL) {
        int size = graph[r*cols + c]->neighbors.size();
        for (int i = 0; i < size; i++) {
          hole_node_edge e(cur, cur->neighbors[i]);
          pair<hole_node_edge, int> tmp1(e, 0);
          pair<hole_node_edge, int> tmp2(e, 1);
          f.insert(tmp1);
          cap.insert(tmp2);
        }
      }
    }
  }

  hole_node * s = new hole_node(-1, -1, -1, -1, 0, -1);
  hole_node * t = new hole_node(-1, -2, -1, -2, 0, -1);
  s->id = -1;
  t->id = -2;
  for (int i = 0; i < srcs.size(); i++) {
    hole_node_edge fwd(s, srcs[i]);
    hole_node_edge bak(srcs[i], s);
    s->neighbors.push_back(srcs[i]);
    srcs[i]->neighbors.push_back(s);
    f.insert(pair<hole_node_edge, int>(fwd, 0));
    f.insert(pair<hole_node_edge, int>(bak, 0));
    cap.insert(pair<hole_node_edge, int>(fwd, 8));
    cap.insert(pair<hole_node_edge, int>(bak, 0));
  }
  for (int i = 0; i < tgts.size(); i++) {
    hole_node_edge fwd(t, tgts[i]);
    hole_node_edge bak(tgts[i], t);
    t->neighbors.push_back(tgts[i]);
    tgts[i]->neighbors.push_back(t);
    f.insert(pair<hole_node_edge, int>(fwd, 0));
    f.insert(pair<hole_node_edge, int>(bak, 0));
    cap.insert(pair<hole_node_edge, int>(fwd, 0));
    cap.insert(pair<hole_node_edge, int>(bak, 8));
  }
  while (residual_path(s, t, rows, cols, &cap, &f, &p)) {
    vertex_hashmap::const_iterator cur_it = p.find(t);
    while (cur_it != p.end()) {
      hole_node * u = cur_it->first;
      hole_node * v = cur_it->second;
      int fwd_flow = (f.find(hole_node_edge(u, v)))->second;
      int bck_flow = (f.find(hole_node_edge(v, u)))->second;
      f[hole_node_edge(v, u)] = fwd_flow + 1;
      f[hole_node_edge(u, v)] = bck_flow - 1;
      cur_it = p.find(cur_it->second);
    }
    p.clear();
  }
  vector<hole_node*> cut = residual_reachable(s, &cap, &f, rows, cols);
  return cut;
}
vector<hole_node*> find_interior(hole_node* root, int prev_width=0) {
  vector<hole_node*> interior;
  if (root->width >= prev_width) {
    for (int i = 0; i < root->children.size(); i++) {
      vector<hole_node*> tmp = find_interior(root->children[i], root->width);
      for (int j = 0; j < tmp.size(); j++) {
        interior.push_back(tmp[j]);
      }
    }
  } else {
    interior.push_back(root);
  }
  return interior;  
}
vector<hole_node*> find_exterior(hole_node** graph, Rect bounding_box) {
  vector<hole_node*> exterior;
  for (int r = 0; r < bounding_box.height; r += bounding_box.height-1) {
    for (int c = 0; c < bounding_box.width; c++) {
      hole_node * cur = graph[r*bounding_box.width + c];
      if (cur != NULL) {
        exterior.push_back(cur);
      }
    }
  }
  for (int r = 0; r < bounding_box.height; r++) {
    for (int c = 0; c < bounding_box.width; c += bounding_box.width-1) {
      hole_node * cur = graph[r*bounding_box.width + c];
      if (cur != NULL) {
        exterior.push_back(cur);
      }
    }
  }

  // if (!bounding_box.contains(Point(root->x, root->y))) {
  //   exterior.push_back(root);
  // } else {
  //   for (int i = 0; i < root->children.size(); i++) {
  //     vector<hole_node*> tmp = find_exterior(root->children[i], bounding_box);
  //     for (int j = 0; j < tmp.size(); j++) {
  //       exterior.push_back(tmp[j]);
  //     }
  //   }
  // }
  return exterior;  
}
vector<hole_node*> find_cut_edge(vector<hole_node*>* cut, hole_node ** graph,
                                 int rows, int cols) {
  int type[rows][cols];
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      if (graph[r*cols+c] != NULL)
        type[r][c] = 1;
      else
        type[r][c] = 0;
    }
  }
  for (int i = 0; i < cut->size(); i++) {
    type[cut->at(i)->r][cut->at(i)->c] = 2;
  }
  vector<hole_node*> edge;
  for (int r = 0; r < rows-1; r++) {
    for (int c = 0; c < cols-1; c++) {
      if (type[r][c] == 2) {
        for (int rp = r; rp <= r + 1; rp++) {
          for (int cp = c; cp <= c + 1; cp++) {
            if ((rp != r || cp != c) && type[rp][cp] == 1) {
              edge.push_back(graph[rp*cols+cp]);
              edge.push_back(graph[r*cols+c]);
            }
          }
        }
      }
    }
  }
  return edge;
}
Mat close_holes(Mat frame, Point center, Rect bounding_box) {
  queue<hole_node*> curq, nextq;
  hole_node* graph[bounding_box.width*bounding_box.height], *start = NULL;
  bool discovered[bounding_box.height][bounding_box.width];
  //cout << "making graph" << endl;
  for (int c = 0; c < bounding_box.width; c++) {
    for (int r = 0; r < bounding_box.height; r++) {
      if (frame.at<uchar>(r + bounding_box.tl().y, c + bounding_box.tl().x) == 255)  {
        graph[r*bounding_box.width + c] = new hole_node(c + bounding_box.tl().x, 
                                                r + bounding_box.tl().y, 
                                                r, c,
                                                bounding_box.width, 0);
        if (c == center.x - bounding_box.tl().x &&
            r == center.y - bounding_box.tl().y) {
          start = graph[r*bounding_box.width + c];
        }
      }
      else {
        graph[r*bounding_box.width + c] = NULL;
        if (c == center.x - bounding_box.tl().x &&
            r == center.y - bounding_box.tl().y) {
          cout << "given center is invalid" << endl;
          bool updated= false;
          for (int rp = r+1; rp < bounding_box.height; rp++) {
            if (frame.at<uchar>(rp + bounding_box.tl().y, c + bounding_box.tl().x) == 255)  {
              center = Point(c + bounding_box.tl().x, rp + bounding_box.tl().y);
              updated = true;
              break;
            }
          }
          if (!updated) return frame;
        }
      }
      discovered[r][c] = false;
    }
  }

  start->width = 0;
  curq.push(start);
  discovered[start->r][start->c] = true;
  int max_dst = 0;
  int max_width = 0;
  //cout << "inital bfs" << endl;
  while (curq.size() > 0) {
    hole_node *cur = curq.front();
    curq.pop();
    if (cur->d > max_dst) max_dst = cur->d;
    for (int c = cur->c-1; c <= cur->c + 1; c += 1) {
      for (int r = cur->r-1; r <= cur->r + 1; r += 1) {
        //if (abs(c-cur->c) + abs(r-cur->r) < 2) {
          if (c >= 0 && c < bounding_box.width && r >= 0 && r < bounding_box.height
              && (c != cur->c || r != cur->r)
              && frame.at<uchar>(r+bounding_box.tl().y, c+bounding_box.tl().x) == 255)  {
            hole_node *node = graph[r*bounding_box.width + c];
            if (discovered[r][c] == false) {
              nextq.push(node);
              discovered[r][c] = true;
              cur->children.push_back(node);
            }
            cur->neighbors.push_back(node);
          }
          //}
      }
    }
    if (curq.size() == 0 && nextq.size() > 0) {
      int width = nextq.size();
      if (width > max_width) max_width = width;
      while (nextq.size() > 0) {
        hole_node *hn = nextq.front();
        nextq.pop();
        hn->width = width;
        curq.push(hn);
      }
    }
  }
  Mat dist = frame.clone();
  //cout << "computing exterior and interior" << endl;
  vector<hole_node*> interior = find_interior(start);
  vector<hole_node*> exterior = find_exterior(graph, bounding_box);
  //cout << interior.size() << endl;
  //cout << exterior.size() << endl;

  /*Mat tmp(dist.size(), CV_8UC3, Scalar(255, 255, 255));
  for (int i = 0; i < interior.size(); i++) {
    hole_node pt = *(interior[i]);

    tmp.at<Vec3b>(pt.y, pt.x) = Vec3b(0, 255, 0);
  }
  for (int i = 0; i < exterior.size(); i++) {
    hole_node pt = *(exterior[i]);
    tmp.at<Vec3b>(pt.y, pt.x) = Vec3b(0, 0, 255);
  }
  for (int i = 0; i < frame.rows; i++) {
    for (int j = 0; j < frame.cols; j++) {
      if (frame.at<uchar>(i, j) == 0) 
        tmp.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
    }
  }


  namedWindow("close holes", CV_WINDOW_AUTOSIZE);
  imshow("close holes", tmp);
  waitKey(0);
  cout << "computing cut" << endl;*/
  vector<hole_node*> cut = min_cut(interior, exterior, graph, 
                                   bounding_box.height, bounding_box.width);
  vector<hole_node*> cut_edge = find_cut_edge(&cut, graph, 
                                              bounding_box.height, bounding_box.width);
  //cout << cut.size() << endl;
  //cout << cut_edge.size() << endl;
  /*for (int i = 0; i < cut.size(); i++) {
    hole_node pt = *(cut[i]);
    if (pt.id >= 0)
      tmp.at<Vec3b>(pt.y, pt.x) = Vec3b(255, 0, 0);
  }
  imshow("close holes", tmp);
  waitKey(0);*/
  for (int i = 0; i < cut_edge.size(); i++) {
    hole_node pt = *(cut_edge[i]);
    dist.at<uchar>(pt.y, pt.x) = (uchar)(0);
  }
  /*double scale = 255.0/((double)start->size);
  cout << scale << " " << start->size << endl;
  int size = discovered.size(), count = 0;
  for (set<hole_node*, bool (*)(hole_node*, hole_node*)>::iterator it = discovered.begin();
       it != discovered.end(); ++it) {
    hole_node *cur = *it;
    int value = (int)(scale*((double)(cur->size)));//(int)min(cur->width, 255);
    dist.at<Vec3b>(cur->y, cur->x) = Vec3b(value, value, value);
    count++;
  }*/
  return dist;
}
vector<Mat> mtlib::filter_debug(Mat& dst, Mat frame) {
  int lowThreshold = 75;
  int ratio = 3;
  dst.create(frame.rows, frame.cols, CV_8UC1);
  vector<Mat> output;
  if (DEF_CHANNEL >= 0 && DEF_CHANNEL < 3) {
    cout << "removing blue" << endl;
    vector<Mat> rgb;
    split(frame, rgb);
    rgb[DEF_CHANNEL].setTo(0);
    merge(rgb, frame);
    output.push_back(frame.clone());
    Mat gray;
    cvtColor(frame, gray, CV_BGR2GRAY);
    Mat gray_color;
    cvtColor(gray, gray_color, CV_GRAY2BGR);
    output.push_back(gray_color.clone());
    cout << "thresholding" << endl;
    //blur(gray, gray, Size(3, 3));
    adaptiveThreshold(gray, dst, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 27, 5);
    Mat dst2;
    adaptiveThreshold(gray, dst2, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 27, 5);
    //GaussianBlur(dst, dst, Size(7, 7), 0, 0);
    Mat dst_color;
    cout << "eroding" << endl;
    cvtColor(dst, dst_color, CV_GRAY2BGR);
    output.push_back(dst_color.clone());
    cvtColor(dst2, dst_color, CV_GRAY2BGR);
    output.push_back(dst_color.clone());
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    erode(dst, dst, element);
    //dilate(dst, dst, element);
    /*namedWindow("Select Center", CV_WINDOW_AUTOSIZE);
    imshow("Select Center", dst);
    vector<Point> pts;
    getNPoints(3, "Select Center", &pts, dst);*/
    //blur(dst, dst, Size(7, 7));
    threshold(dst, dst, 40, 255, THRESH_BINARY);
    cout << "closing holes" << endl;
    Mat closed = close_holes(dst, Point(308, 293), Rect(Point(252, 222), Point(377, 373)));
    cout << "closed" << endl;
    Mat closed_color;
    cvtColor(closed, closed_color, CV_GRAY2BGR);
    output.push_back(closed_color.clone());
    return output;
  } else {
    cout << "WARNING: using an defuct def_channel" << endl;
    Mat t1, t2;
    cvtColor(frame, t1, CV_BGR2GRAY);
    Canny(t1, t1, lowThreshold, lowThreshold*ratio, 5);
    adaptiveThreshold(t1, dst, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 27, 5);
    vector<Mat> tmp;
    return tmp;
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
void mtlib::closeHolesAndFindContours(Mat frame, vector< vector<Point> > * contours, 
                                      Point off, Point center, Rect bb)
{
  Mat t;
  filter(t, frame);
  Mat t2 = close_holes(t, center, bb);
  vector<Vec4i> h;
  Mat t3;
  bitwise_not(t2, t3);
  findContours(t3, *contours, h, CV_RETR_LIST, CV_CHAIN_APPROX_NONE, off);
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
void mtlib::drawContoursBoxed(Mat dst, vector< vector<Point> > * contours, 
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
      Point2f verticies[4];
      minAreaRect(contour).points(verticies);
      for (int i = 0; i < 4; i++)
        line(dst, verticies[i], verticies[(i+1)%4], Scalar(255, 0, 0), 2);
    }
  }
 
}
void mtlib::drawContoursAndFilter(Mat dst, vector< vector<Point> > * contours, 
                                  int minArea, int maxArea)
{
  Mat contour_drawing = Mat::zeros(dst.size(), dst.type());

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

  RotatedRect rr = getBoundingBox(t);
  double a = rr.angle;
  Point2f ihat(std::cos(a*PI/180), -std::sin(-a*PI/180));
  Point2f jhat(-ihat.y, ihat.x);

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
    {case MFILLED:
        vector< vector<Point> > contours;
        contours.push_back(cont);
        drawContours(dst, contours, 0, Scalar(255, 255, 255), CV_FILLED);
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
        idir = 0;
        jdir = 0;
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
        idir = 0;
        jdir = 0;
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
Rect combine_rectangles(Rect r1, Rect r2) {
  Point tl(min(r1.tl().x, r2.tl().x), min(r1.tl().y, r2.tl().y));
  Point br(max(r1.br().x, r2.br().x), max(r1.br().y, r2.br().y));
  return Rect(tl, br);
}
Rect resize_rectangle(Rect r, double scale, Point tl, Point br) {
  int nw = r.width*scale, nh = r.width*scale;
  Point shift((nw-r.width)/2, (nh-r.height)/2);
  Point ntl = r.tl() - shift;
  if (ntl.x < tl.x) ntl.x = tl.x;
  if (ntl.y < tl.y) ntl.y = tl.y;
  Point nbr = r.br() + shift;
  if (nbr.x >= br.x) ntl.x = br.x-1;
  if (ntl.y >= br.y) ntl.y = br.y-1;

  return Rect(r.tl() - shift, r.br() + shift);
}
int best_contour_index(vector< vector<Point> > contours, mtlib::Model * m, 
                       int minArea, int maxArea) {
  int bestCont = 0, area_diff = abs(contourArea(contours[0]) - m->getArea());
  bool foundObject = false;
  cout << "looping over contours" << endl;
  for (int n = 0; n < contours.size(); n++) {
    double consize = contourArea(contours[n]);

    int consize_diff = abs(consize - m->getArea());
    if (consize > minArea && consize < maxArea && consize_diff <= area_diff) {
      cout << "found new candidate" << endl;
      foundObject = true;
      bestCont = n;
      area_diff = consize_diff;
    }
  }
  if (!foundObject) return -1;
  cout << "found object" << endl;
  return bestCont;
}

void mtlib::updateModels(Mat frame, vector<Model> * models, int minArea, int maxArea,
                         bool close_holes, unsigned long timestamp) {
  //loop through models
  Mat out = Mat::zeros(frame.size(), frame.type());
  //namedWindow("Searching", CV_WINDOW_AUTOSIZE);
  unsigned long t = time_milli();

  for (int n = 0; n < models->size(); n++) {
    cout << "Updating model " << n << " with area " << models->at(n).getArea() << endl;
    //Get part of image in search area
    Rect searchArea = models->at(n).getSearchArea(frame);
    Mat roi(frame.clone(), searchArea);
    
    //do contour finding and filtering
    vector< vector<Point> > contours, tmp;
    filterAndFindContours(roi, &contours, searchArea.tl());
    
    for (int i = 0; i < contours.size(); i++) {
      double consize = contourArea(contours[i]);
      if (consize >= minArea  && consize <= maxArea) {
        tmp.push_back(contours[i]);
      }
    }
    contours = tmp;
    if (contours.size() > 0) {
      bool foundObject = true;
      int bestCont = best_contour_index(contours, &(models->at(n)), minArea, maxArea);
      if (bestCont < 0) foundObject = false;
      double err = ((double)(abs(contourArea(contours[bestCont])-models->at(n).getArea())))
                    /models->at(n).getArea();
      if (close_holes && err > 0.5) {
        cout << "Model not found, attempting to close holes. Error: " << err << endl;
        // cout << "sorting " << contours.size() << " contours" << endl;
        // for (int i = 0; i < contours.size(); i++) {
        //   for (int j = 0; j < contours.size()-1; j++) {
        //     if (contourArea(contours[j]) < contourArea(contours[j+1])) {
        //       vector<Point> tmp = contours[j+1];
        //       contours[j+1] = contours[j];
        //       contours[j] = tmp;
        //     }
        //   }
        // }
        // cout << "done sorting" << endl;
        // Rect bounding_box = boundingRect(contours[0]);
        // double area_sum = contourArea(contours[0]);
        // Point center = area_sum*getCenter(contours[0]);
        // cout << "finding bounding box" << endl;
        // for (int i = 1; i < contours.size(); i++) {
        //   double err = ((double)(abs(contourArea(contours[i])-models->at(n).getArea())))
        //     /models->at(n).getArea();          
        //   if (err > 0.1) {
        //     bounding_box = combine_rectangles(bounding_box, boundingRect(contours[i]));
        //     double consize = contourArea(contours[i]);
        //     Point c = getCenter(contours[i]);
        //     center = center + Point(c.x*consize, c.y*consize);
        //     area_sum += consize;
        //   }
        // }
        // center = Point(center.x/area_sum, center.y/area_sum);
        // bounding_box = resize_rectangle(bounding_box, 1.1, searchArea.tl(), searchArea.br());
        Rect bounding_box = resize_rectangle(searchArea, 0.8, searchArea.tl(), searchArea.br());
        Point center = models->at(n).getCenter() - searchArea.tl();
        bounding_box = Rect(bounding_box.tl() - searchArea.tl(), bounding_box.size());
        contours.clear();
        cout << "closing " << roi.size() << " " << center << " " << bounding_box << endl;
        closeHolesAndFindContours(roi, &contours, searchArea.tl(), center,
                                  bounding_box);
        bestCont = best_contour_index(contours, &(models->at(n)), minArea, maxArea);
        if (bestCont < 0) foundObject = false;
      }
      Point2d rot_info;
      Point c;
      if (foundObject) {
        c = getCenter(contours[bestCont]);
        rot_info = models->at(n).getContourRot(contours[bestCont], c);
        cout << "Angle: " << rot_info.x << " Cost: " << rot_info.y << endl;
        if (rot_info.y > 200) {
          cout << "Cost too high, model not found" << endl;
          foundObject = false;
        }
        //if (rot_info.y > 1000) foundObject = false;
      }

      cout << "Found object: " << std::boolalpha << foundObject << endl;
      //if the object was found generate new center and orientation data
      //othrewise assume it hasn't moved

      double a, cost;
      vector<double> r;
      vector<Point> cont;
      
      if (foundObject) {

        r = getRotSignal(contours[bestCont], c);
        //showHist("Histogram", sig);
        //ap = getRotation(models->at(i), roi_cont, 45);

        a = rot_info.x;
        cost = rot_info.y;
        //cout << ap << ", " << a << endl;
        cont = contours[bestCont];
      } else {
        c = models->at(n).getCenter();
        a = models->at(n).getRotation();
        r = models->at(n).getRotationSignal();
        cont = models->at(n).getContour();
        cost = -1;
      }


      //circle(frame, c, 4, Scalar(255, 0, 0), -1, 8, 0);
      //imshow("Found", frame);
      //waitKey(0);
      models->at(n).update(c, a, r, cont, foundObject, timestamp, cost);
      Mat tmp_frame(frame.rows, frame.cols, CV_8UC3, Scalar(0, 0, 0));
      models->at(n).drawContour(tmp_frame);
      models->at(n).drawBoundingBox(tmp_frame, Scalar(255, 0, 0));
      Mat tmp_roi(tmp_frame, searchArea);
    } else {
      cout << "Found no contours" << endl;
      Point c = models->at(n).getCenter();
      double a = models->at(n).getRotation();
      vector<double> r = models->at(n).getRotationSignal();
      vector<Point> cont = models->at(n).getContour();
      models->at(n).update(c, a, r, cont, false, timestamp, -1); 
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
  m->drawBoundingBox(frame, color, 0);
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
Point2f get_light_center(Mat frame, int thresh, int bar) {
  Point2f tot(0, 0);
  int cnt = 0;
  for (int r = bar; r < frame.rows-bar; r++) {
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
vector<Point2f> mtlib::autoCalibrate(long (*capture)(Mat*), string dmd_window, Size dmd_size) {
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

      c = get_light_center(gray, thresh, 40);
      threshold(gray, thresh_img, thresh, 255, THRESH_BINARY);
      rectangle(thresh_img, Point(0, 0), Point(thresh_img.cols, 40), Scalar(0, 0, 0), CV_FILLED);
      rectangle(thresh_img, Point(0, thresh_img.rows-40), 
                Point(thresh_img.cols, thresh_img.rows), Scalar(0, 0, 0), CV_FILLED);
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

bool mtlib::file_exists(const std::string name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}
string mtlib::safe_filename(char * prefix, const char * suffix) {
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
  sprintf(buffer, "%s-%.6d%s", filename, count, suffix);
  while (file_exists(buffer)) {
    count++;
    sprintf(buffer, "%s-%.6d%s", filename, count, suffix);
  }
  cout << "Writing frame " << buffer << endl;
  imwrite(buffer, frame);
}

void mtlib::set_output_figure_name (string name) {
  output_figure_filename = name;
}



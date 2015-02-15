#include "stdafx.h"
#include <iostream>
//#include "core/core.hpp"
//#include "core/core_c.h"
#include <opencv.hpp>
#include "imgproc/imgproc.hpp"
#include "highgui/highgui.hpp"
#include <iostream>
#include <cstdlib> //for timer
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <fstream>
#include <string> // for strings
#include <iomanip> // for controlling float print precision
#include <sstream> // string to number conversion
#include <time.h>
#include <uEye.h>
#include <uEye_tools.h>
#include "C:\OpenCV2.3\build\include\
opencv\cv.h"
#include "C:\OpenCV2.3\build\include\opencv\highgui.h"
#include "C:\OpenCV2.3\build\include\opencv\ml.h"

#define NULL 0

using namespace std;
using namespace cv;
using namespace cv::flann;


void on_mouse( int e, int x, int y, int d, void *ptr )
{
    Point*p = (Point*)ptr;
    p->x = x;
    p->y = y;
}

cv::Mat framew;
Mat img1; 
Mat white;
Mat img_bw;
Mat gray_img;
Mat flip_img;
Mat flip_img_crop;
Mat img_bw_crop;
int n;
float srcx1;
float srcy1;
float srcx2;
float srcy2;
float srcx3;
float srcy3;
float dstx1;
float dsty1;
float dstx2;
float dsty2;
float dstx3;
float dsty3;

Point p;
Point2f srcTri[3];
Point2f dstTri[3];

vector< vector<Point> > contours;
vector<Vec4i> hierarchy;

int main(){

        /////////////////////////////////////////////////////////
    //This is used to set the video parameters: these functions are very specific for the ueye usb camera-They will not work for any other camera
    //Create an image in the c image format
    IplImage* fram =NULL;
    IplImage *fram1=NULL;
    //Set the usb camera to slot 1
    HIDS hCam = 1;
    //Set the video parameters
    IS_SIZE_2D is_size;
    is_size.s32Width = 752;
    is_size.s32Height = 480;

    INT nRet = is_AOI( hCam, IS_AOI_IMAGE_SET_AOI, (void*)&is_size, sizeof(is_size));
    //This is used to set the video parameters: these functions are very specific for the ueye usb camera-They will not work for any other camera
    ///////////////////////////////////////////////////////////

    VideoCapture cap(1); // open the camera in slot 1
    if(!cap.isOpened())  // check if we succeeded
    {
        return -1;
    }
    VideoWriter outputVideo; 
    outputVideo.open("dmdvid.avi",CV_FOURCC('C','V','I','D'),15,Size(752,480),1);
    if (!outputVideo.isOpened())
    {
        cout  << "Could not open the output video for write.";
        return -1;
    }

    //Start image capture process. Not sure why this is necessary first, but a white screen won't display by other methods.
    namedWindow("ueye", CV_WINDOW_NORMAL);
    //    cvResizeWindow("ueye", 752, 480);

    cout << "New affine transform or reuse old (1/2).";
    int affine;
    cin >> affine;
    cout << "The value you entered is " << affine << ".\n";

    if (affine==1){
    cout << "Here's the first frame.\n";

    cap>>framew;
    flip(framew, flip_img, 1); 
    imshow("ueye", flip_img);


    // Collect three source points for affine transformation.
    cout << "Click first point for transformation and hit a key.\n";
    setMouseCallback("ueye",on_mouse, (void*)(&p) ); // Retrieve first point for affine transformation
    waitKey(0);
    srcTri[0] = Point2f(p.x,p.y);
    cout << "Mouse coordinates - " << srcTri[0] << "\n" ;
    srcx1 = p.x;
    srcy1 = p.y;

    cout << "Click second point for transformation and hit a key.\n";
    setMouseCallback("ueye",on_mouse, (void*)(&p) ); // Retrieve first point for affine transformation
    waitKey(0);
    srcTri[1] = Point2f(p.x,p.y);
    cout << "Mouse coordinates - " << srcTri[1] << "\n" ;
    srcx2 = p.x;
    srcy2 = p.y;

    cout << "Click third point for transformation and hit a key.\n";
    setMouseCallback("ueye",on_mouse, (void*)(&p) ); // Retrieve first point for affine transformation
    waitKey(0);
    srcTri[2] = Point2f(p.x,p.y);
    cout << "Mouse coordinates - " << srcTri[2] << "\n" ; 
    srcx3 = p.x;
    srcy3 = p.y;

    outputVideo << framew;
    //    imwrite( "C:/Documents and Settings/Administrator/My Documents/Visual Studio 2008/Projects/DLP_Control3/desktop_gears.jpg", flip_img);
    //img1 = imread("C:/Documents and Settings/Administrator/Desktop/white.jpg"); //image path
    //Save the image as a grayscale image - despite the image appearing grey, there is color information in Mat
    cvtColor(framew, gray_img, CV_BGR2GRAY);

    // Threshhold this initial capture to create a 'white' image. This method is very roundabout.
    threshold(gray_img, white, 255, 255, THRESH_BINARY);

    namedWindow("dmd", CV_WINDOW_NORMAL);
    cvMoveWindow("dmd", 1275, -32);
    cvResizeWindow("dmd", 608, 684);
    imshow("dmd", white);
    waitKey(300); 

//    img1 = imread("C:/Documents and Settings/Administrator/Desktop/white.jpg"); //image path
//    cvtColor(img1, gray_img, CV_BGR2GRAY);
//    threshold(gray_img, white, 255, 255, THRESH_BINARY);

    //Display the white image on the DMD


    imshow("dmd", white);
    waitKey(100);

    cap>>framew;

    flip(framew, flip_img, 1);
    imshow("ueye", flip_img);
    waitKey(30);
    
    cap>>framew;  // This line is essential to keep the video from 'feeding back'.  Timing issue?

    cvtColor(framew, gray_img, CV_BGR2GRAY);
    adaptiveThreshold(gray_img, img_bw, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 27, 5);

    // Crop the full image to that image contained by the rectangle myROI
    //Rect myROI(90, 50, 630, 350);
    Rect myROI(1, 1, 640, 479);
    img_bw_crop = img_bw(myROI).clone();

    //Display the threshholded image on DMD
    flip(img_bw_crop, flip_img_crop, 0);
    imshow("dmd", flip_img_crop);
    waitKey(1000);

 //  Loop over display of camera video.  Not sure why it's necessary for the 'delay'
 n=1;
 while(n<3)
 {
    cout << "Display ueye image for second part of affine transformation " << n << "\n" ; 
    //Grab a frame from the camera
    cap>>framew;

    //Display the captured image on screen
    flip(framew, flip_img, 1);
    imshow("ueye", flip_img);
    waitKey(100);
    outputVideo << framew;
    n++;
  }
 
    cout << "Click fourth point for transformation and hit a key.\n";
    setMouseCallback("ueye",on_mouse, (void*)(&p) ); // Retrieve first point for affine transformation
    waitKey(0);
    dstTri[0] = Point2f(p.x,p.y);
    cout << "Mouse coordinates - " << dstTri[0] << "\n" ; 
    dstx1 = p.x;
    dsty1 = p.y;

    cout << "Click fifth point for transformation and hit a key.\n";
    setMouseCallback("ueye",on_mouse, (void*)(&p) ); // Retrieve first point for affine transformation
    waitKey(0);
    dstTri[1] = Point2f(p.x,p.y);
    cout << "Mouse coordinates - " << dstTri[1] << "\n" ;
    dstx2 = p.x;
    dsty2 = p.y;

    cout << "Click sixth point for transformation and hit a key.\n";
    setMouseCallback("ueye",on_mouse, (void*)(&p) ); // Retrieve first point for affine transformation
    waitKey(0);
    dstTri[2] = Point2f(p.x,p.y);
    cout << "Mouse coordinates - " << dstTri[2] << "\n" ; 
     dstx3 = p.x;
    dsty3 = p.y;

    // Save coordinates to file to skip calibration step.
    ofstream myfile;
    myfile.open ("affine_transform_values.txt");
    myfile << srcx1 << endl;
    myfile << srcy1 << endl;
    myfile << srcx2 << endl;
    myfile << srcy2 << endl;
    myfile << srcx3 << endl;
    myfile << srcy3 << endl;
    myfile << dstx1 << endl;
    myfile << dsty1 << endl;
    myfile << dstx2 << endl;
    myfile << dsty2 << endl;
    myfile << dstx3 << endl;
    myfile << dsty3 << endl;
    myfile.close();

    }
    else
    {
    cout << "Reading in old affine transform data.\n";
    std::fstream myfile("affine_transform_values.txt", std::ios_base::in);
    myfile >> srcx1 >> srcy1 >> srcx2 >> srcy2 >> srcx3 >> srcy3 >> dstx1 >> dsty1 >> dstx2 >> dsty2 >> dstx3 >> dsty3;

    srcTri[0] = Point2f(srcx1,srcy1);
    srcTri[1] = Point2f(srcx2,srcy2);
    srcTri[2] = Point2f(srcx3,srcy3);
    dstTri[0] = Point2f(dstx1,dsty1);
    dstTri[1] = Point2f(dstx2,dsty2);
    dstTri[2] = Point2f(dstx3,dsty3);

    // Add code to get an image to work with.
    cout << "Here's the first frame.\n";

    cap>>framew;
    flip(framew, flip_img, 1); 
    imshow("ueye", flip_img);

    outputVideo << framew;
    //    imwrite( "C:/Documents and Settings/Administrator/My Documents/Visual Studio 2008/Projects/DLP_Control3/desktop_gears.jpg", flip_img);
    //img1 = imread("C:/Documents and Settings/Administrator/Desktop/white.jpg"); //image path
    //Save the image as a grayscale image - despite the image appearing grey, there is color information in Mat
    cvtColor(framew, gray_img, CV_BGR2GRAY);

    // Threshhold this initial capture to create a 'white' image. This method is very roundabout.
    threshold(gray_img, white, 255, 255, THRESH_BINARY);

    namedWindow("dmd", CV_WINDOW_NORMAL);
    cvMoveWindow("dmd", 1275, -32);
    cvResizeWindow("dmd", 608, 684);
    imshow("dmd", white);
    waitKey(300); 

    //Display the white image on the DMD

    imshow("dmd", white);
    waitKey(100);

    cap>>framew;

    flip(framew, flip_img, 1);
    imshow("ueye", flip_img);
    waitKey(30);
    
    cap>>framew;  // This line is essential to keep the video from 'feeding back'.  Timing issue?

    cvtColor(framew, gray_img, CV_BGR2GRAY);
    adaptiveThreshold(gray_img, img_bw, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 7, 5);

    // Crop the full image to that image contained by the rectangle myROI
    //Rect myROI(90, 50, 630, 350);
    Rect myROI(1, 1, 640, 479);
    img_bw_crop = img_bw(myROI).clone();

    //Display the threshholded image on DMD
    flip(img_bw_crop, flip_img_crop, 0);
    imshow("dmd", flip_img_crop);
    waitKey(1000);
    }

    Mat warp_mat( 2, 3, CV_32FC1 );
    Mat warp_dst, warp_rotate_dst;
    /// Set the dst image the same type and size as src
    warp_dst = Mat::zeros( flip_img_crop.rows, flip_img_crop.cols, flip_img_crop.type() );
    /// Get the Affine Transform
    warp_mat = getAffineTransform( dstTri, srcTri );
    /// Apply the Affine Transform just found to the src image
    warpAffine( flip_img_crop, warp_dst, warp_mat, warp_dst.size(), INTER_LINEAR, BORDER_CONSTANT, 255 );


    //Display the warped, threshholded image on DMD
    //flip(img_bw_crop, flip_img_crop, 0);
    imshow("dmd", warp_dst);
    waitKey(1000);

 //  Loop over display of camera video.  Not sure why it's necessary for the 'delay'
 n=1;
 while(n<3)
 {
    cout << "Display ueye image for warped dmd image " << n << "\n" ; 
    //Grab a frame from the camera
    cap>>framew;

    //Display the captured image on screen
    flip(framew, flip_img, 1);
    imshow("ueye", flip_img);
    waitKey(100);
    outputVideo << framew;
    waitKey(500);
    n++;
  }

    cout << "Hit a key to stop video capture.\n";

    while(true)
    {
    // Background gray level (can adjust to 255 for black)
    rectangle( img_bw_crop, Point( 1, 1), Point( 640, 479), Scalar( 1, 1, 1 ),-1,8 );
    imshow("dmd", img_bw_crop);
    //Display the white image on the DMD
//    imshow("dmd", white);
    waitKey(100);

    cap>>framew;

    flip(framew, flip_img, 1);
    imshow("ueye", flip_img);
    waitKey(30);
    
    cap>>framew;  // This line is essential to keep the video from 'feeding back'.  Timing issue?

    cvtColor(framew, gray_img, CV_BGR2GRAY);
    adaptiveThreshold(gray_img, img_bw, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 7, 5);

    // Crop the full image to that image contained by the rectangle myROI
    //Rect myROI(90, 50, 630, 350);
    Rect myROI(1, 1, 640, 479);
    img_bw_crop = img_bw(myROI).clone();

    //Display the threshholded image on DMD
    flip(img_bw_crop, flip_img_crop, 0);
    warpAffine( flip_img_crop, warp_dst, warp_mat, warp_dst.size(), INTER_LINEAR, BORDER_CONSTANT, 255 );
    imshow("dmd", warp_dst); // was flip_img_crop
    waitKey(30);

    n=1;
     while(n<10)
    {
    cout << "Display ueye - " << n << "\n" ; 
    //Grab a frame from the camera
    cap>>framew;

    //Display the captured image on screen
    flip(framew, flip_img, 1);
    imshow("ueye", flip_img);
    waitKey(10);
    outputVideo << framew;
    n++;

    //findContours(img_bw, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
    /*Mat blue = Mat::zeros( 518, 678, CV_8UC3 );

        for( int i = 0; i< contours.size(); i++ )
            {
            //      printf(" * Contour[%d] - Area OpenCV: %.2f \n", i,
            contourArea(contours[i]) );
            Scalar color = Scalar( 255, 0, 0 );
            float consize = contourArea(contours[i]);
            if(consize>500.0){
                drawContours( blue, contours, i, color, 10, 8,
                hierarchy, 0, Point() );
            }
        }*/
    }
    if(waitKey(100) >= 0) break;        
    }
    return 0;
}

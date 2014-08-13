#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>
#include<cstdio>

using namespace cv;
using namespace std;

int erosion_type = MORPH_RECT;
int erosion_size = 4;

MatND histogram(Mat src);
 
int main(int argc, char *argv[])
{
    Mat frame,frameHsv;
    Mat back;
    Mat fore;
Mat mask,fleshMask,fleshMask2,fleshMask3;
Mat labels;
Mat centers,centers8,centersImg,centersHsv,centersHsvFiltered;
Mat kframe, kframe32;
Mat outframe,outframeScaled;

int colourPoints = 5;
int lights = 640;

    VideoCapture cap(0);
    BackgroundSubtractorMOG2 bg;

    bg.set("nmixtures",4);
    bg.set("detectShadows",true);

//namedWindow("Values");
int h1min = 0, h1max = 20;
int h2min = 127, h2max = 255;
int smin = 0, smax = 114;
int vmin = 68, vmax = 207;
createTrackbar("H1min", "Values", &h1min, 255);
createTrackbar("H1max", "Values", &h1max, 255);
createTrackbar("H2min", "Values", &h2min, 255);
createTrackbar("H2max", "Values", &h2max, 255);
createTrackbar("Smin", "Values", &smin, 255);
createTrackbar("Smax", "Values", &smax, 255);
createTrackbar("Vmin", "Values", &vmin, 255);
createTrackbar("Vmax", "Values", &vmax, 255);


Mat element = getStructuringElement( erosion_type,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) );
 
    std::vector< std::vector<Point> > contours;
vector<Vec4i> hierarchy;

	CvTermCriteria criteria;
	criteria.type = CV_TERMCRIT_ITER + CV_TERMCRIT_EPS;
	criteria.max_iter = 10;
	criteria.epsilon = 0.1;

outframe = Mat::zeros(1, colourPoints, CV_8UC3);
outframeScaled = Mat::zeros(100, lights, CV_8UC3);

    while(true)
    {
        cap >> frame;

// Downsize to save CPU
resize(frame, frame, Size(frame.cols / 4, frame.rows / 4));

cvtColor(frame, frameHsv, CV_BGR2HSV); 

inRange(frameHsv, Scalar(h1min,smin,vmin), Scalar(h1max,smax,vmax), fleshMask);
inRange(frameHsv, Scalar(h2min,smin,vmin), Scalar(h2max,smax,vmax), fleshMask2);
fleshMask = ~ (fleshMask | fleshMask2);
cvtColor(fleshMask, fleshMask, CV_GRAY2RGB);
frame = frame.mul(fleshMask);

        bg.operator ()(frame,fore);

bg.getBackgroundImage(back);

        //erode(fore,fore,element);
        //dilate(fore,fore,element);
        findContours(fore,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);

// Create the mask from the 'moving' components
for(int c = 0; c < contours.size(); c++) {
	if(contourArea(contours[c]) > 100.0) {

		mask = Mat::zeros(frame.rows, frame.cols, frame.type());
drawContours(mask,contours,c,Scalar(255,255,255),CV_FILLED);

		// Make our image only comprise of the moving components
		mask &= frame;
mask &= fleshMask;

// Downsize to save CPU when running the kmeans
resize(mask, kframe, Size(10, 10), 0, 0, INTER_NEAREST);
kframe.convertTo(kframe32, CV_32FC3);
kframe32 = kframe32.reshape(1, kframe.rows * kframe.cols);
kmeans(kframe32, 2, labels, criteria, 2, KMEANS_PP_CENTERS, centers);

centers.convertTo(centers8, CV_8U);
centers8 = centers8.reshape(3, 1);
centers8 = centers8.colRange(1, 2);
cvtColor(centers8, centersHsv, CV_RGB2HSV);
centersHsvFiltered = Mat(0, 0, CV_8UC3);
for(int col = 0; col < centersHsv.cols; col++) {
	Vec3b colour = centersHsv.at<Vec3b>(0,col);
	if(colour[2] > 100) {
		colour[1] *= 1.5;
		colour[2] *= 1.5;
		centersHsvFiltered.push_back(colour);
	}
}
if(centersHsvFiltered.rows == 0) continue;
cvtColor(centersHsvFiltered, centers8, CV_HSV2RGB);
resize(centers8, centersImg, Size(1, 1), 0, 0, INTER_NEAREST);
Point p = contours[c].at((0,0));
p.x = (int)((float) p.x / frame.cols * outframe.cols);
if(p.x + centersImg.cols > outframe.cols) p.x = outframe.cols - centersImg.cols - 1; 
p.y = 0;
Rect roi(p, Size(centersImg.rows, centersImg.cols));
Mat frameRoi = outframe(roi);
centersImg.copyTo(frameRoi);
}
}

resize(outframe, outframeScaled, Size(outframeScaled.cols, outframeScaled.rows), 0, 0, INTER_LINEAR);
flip(outframeScaled, outframeScaled, -1);

imshow("Strip", outframeScaled);
imshow("Frame flesh", fleshMask);
imshow("Masked frame", mask);
imshow("frame", frame);
//imshow("Background", back);

        if(waitKey(10) >= 0) break;
    }
    return 0;
}

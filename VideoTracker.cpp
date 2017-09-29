// CppVideoTracker
//The code estimates movement (translation/rotation) from a video file, camera facing towards the floor from 60 cm. It uses OpenCV libraries. The video is separated in frames, keypoints identified, translation-rotation calculated between two frames. The dislocation is visualized in an X-Y graph.
#include <opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include <windows.h>  

//rotation 530-700 frames

using namespace std;
using namespace cv;

int heightCamera = 60; //height in cm
int angleCamera = 85; //degrees
int start, stop;
std::ofstream f("xy.txt"); //result file

struct translation {
	int frame; //no frame or vector, same as reference of vector a[0].frame=0
	int x; //relative disp x
	int y;
	double angle; //angle of trans
	double intensity;
} dummytranslation ; //dummy to collect data before it's written to vector

struct rotation {
	int frame; //no frame
	double angle; //angle of rot
	double overallangle; //sum of angles
} dummyrotation ; //dummy to collect data before it's written to vector

struct displacement {
	int frame; //no frame
	double x; //relative disp, can be fraction due to atan conversions
	double y; //
	double sumx; //absolute disp
	double sumy;
	double cmx; // in cm
	double cmy;
} dummydisplacement ; //dummy to collect data before it's written to vector

vector <translation> transvector; //vectors containing all data
vector <rotation> rotvector; //vectors containing all data
vector <displacement> dispvector; //vectors containing all data
double rotb(0); //for rotation
double a = 57.2957795; //rad to deg
double theta; //Euler angle matrix, two solutions


inline static void allocateOnDemand(IplImage **img, CvSize size, int depth, int channels
)
{
	if (*img != NULL) return;
	*img = cvCreateImage(size, depth, channels);
}

inline static double square(int a) { return a * a; }

int main(void) {
	//The initial part is C language because I couldn't make VideoCapture work (C++ syntax)
	// set input video
	CvCapture *capVideo = cvCaptureFromFile("rotatenew.avi");//read in
	cvQueryFrame(capVideo); //hack, else it wont work
							
							/* Read the video's frame size out of the AVI. */
	CvSize frame_size;
	frame_size.height = (int)cvGetCaptureProperty(capVideo, CV_CAP_PROP_FRAME_HEIGHT);
	frame_size.width = (int)cvGetCaptureProperty(capVideo, CV_CAP_PROP_FRAME_WIDTH);

	//get centerpoint coordinates
	CvPoint centerpoint;
	centerpoint.x = frame_size.width / 2; //x centerpoint
	centerpoint.y = frame_size.height / 2; //y centerpoint

	//Distance calculation
	//Camera calibration would be better
	double pixelCm = heightCamera* tan(angleCamera * 0.5 * 3.14 / 180)/max(centerpoint.x,centerpoint.y); //in cm/pixel

	/* Determine the number of frames in the AVI. */
	long number_of_frames;
	cvSetCaptureProperty(capVideo, CV_CAP_PROP_POS_AVI_RATIO, 1.); //go to end
	number_of_frames = (int)cvGetCaptureProperty(capVideo, CV_CAP_PROP_POS_FRAMES); //get no
	cvSetCaptureProperty(capVideo, CV_CAP_PROP_POS_FRAMES, 0.); //go to start


	long current_frame = 0; //frame counter
	
	//allocate
	IplImage *frame1 = NULL, *frame2 = NULL, *frame1_1C = NULL, *frame2_1C = NULL, *fr = NULL;
	allocateOnDemand(&frame1_1C, frame_size, IPL_DEPTH_8U, 1); //first frame grayscale
	allocateOnDemand(&frame2_1C, frame_size, IPL_DEPTH_8U, 1); //second grayscale
	allocateOnDemand(&frame1, frame_size, IPL_DEPTH_8U, 3); //color image for graph
	allocateOnDemand(&frame2, frame_size, IPL_DEPTH_8U, 3); //color image for graph


	//has to be sent over to every next iteration
	cv::Mat img1, img2, img3, img4; //first two are for comparison the second two for drawing 
	vector<KeyPoint> kpts1, kpts2; //keypoints
	Mat desc1, desc2; //descriptors

	// Create axes displacement
	Point origo(430,415);
	Mat image = Mat::zeros(845, 845, CV_8UC3); 
	Scalar color(0, 255, 255);
	rectangle(image, Point(30, 15), Point(830, 815), color, +1, 4);	// Draw a rectangle
	putText(image,String ("Y"),Point(7,410), FONT_HERSHEY_SIMPLEX,0.5,color,1,8,false);
	putText(image, String("[cm]"), Point(0, 430), FONT_HERSHEY_SIMPLEX, 0.35, color, 1, 8, false);
	putText(image, String("X [cm]"), Point(410, 830), FONT_HERSHEY_SIMPLEX, 0.5, color, 1, 8, false);
	circle(image, origo, 2, color, 1, 8, 0);
	
	for (int i = 0; i < 4; i++)
	{
		line(image, Point(30, 415+i * 100), Point(40, 415+ i * 100), color, 1, 8, 0); 
		line(image, Point(430 + i * 100,815), Point(430 + i * 100,805), color, 1, 8, 0);
		line(image, Point(30, 415 - i * 100), Point(40, 415 - i * 100), color, 1, 8, 0);
		line(image, Point(430 - i * 100, 815), Point(430 - i * 100, 805), color, 1, 8, 0);

		putText(image, to_string(-200 * i), Point(45, 415 + i * 100), FONT_HERSHEY_SIMPLEX, 0.35, color, 1, 8, false);
		putText(image, to_string(200 * i), Point(430 + i * 100,795), FONT_HERSHEY_SIMPLEX, 0.35, color, 1, 8, false);
		putText(image, to_string(200 * i), Point(45, 415 - i * 100), FONT_HERSHEY_SIMPLEX, 0.35, color, 1, 8, false);
		putText(image, to_string(-200 * i), Point(430 - i * 100, 795), FONT_HERSHEY_SIMPLEX, 0.35, color, 1, 8, false);

	}
	// 50 pixel is 100 cm= 0.5 pixel/cm
	Point currentplace(origo),pointtomove(origo),pointfrom(origo);


	//Initialize detectors, choose by uncommenting
	//BRISK/AKAZE many points but slow, ORB fast but few points
	// Akaze handles rotation best
	//Initialize outside to save time

	//BRISK
	//Ptr<BRISK> detector = BRISK::create();
	//akaze
	Ptr<AKAZE> detector = AKAZE::create();
	//orb
	//Ptr<ORB> detector = ORB::create();

	int number_of_steps(2);

	while (true)
	{
		cvSetCaptureProperty(capVideo, CV_CAP_PROP_POS_FRAMES, current_frame); //go to the corresponding frame

		if (current_frame ==0) // only do this when it first runs
		{

			fr = cvQueryFrame(capVideo); // Get first frame of the video.
			/* Convert whatever the AVI image format is into OpenCV's preferred format.
			* AND flip the image vertically. Flip is a shameless hack. OpenCV reads in AVIs upside-down by default.*/
			cvConvertImage(fr, frame1_1C, CV_CVTIMG_FLIP);
			img1 = cv::cvarrToMat(frame1_1C); // grayscale image in mat format
			img3 = cv::cvarrToMat(frame1); //color for drawing
		}
		else //from the second frame use the frames from the previous one
		{
			img1 = img2; //grayscale
			img3 = img4; //color
		}

		//second frame
		for (size_t i = 0; i < number_of_steps-1; i++) cvQueryFrame(capVideo); //skip frames
				
		fr = cvQueryFrame(capVideo); //read in the second frame
		
		
		cvConvertImage(fr, frame2_1C, CV_CVTIMG_FLIP);// Convert and flip
		cvConvertImage(fr, frame2, CV_CVTIMG_FLIP); //full color
		img2 = cv::cvarrToMat(frame2_1C); //image to mat format
		img4 = cv::cvarrToMat(frame2); //color for drawing


			//Images loaded in, the rest is c++ syntax

		//Start stop of frame investigation
		start = 0; //530 for rot
		stop = number_of_frames; //700 for rot, number_of_frames true end
		if (current_frame >= start && current_frame < stop) {
			//end of bracket around line 350


			if (current_frame == start) //only calculate first if its the first execution, saves time
			{
				detector->detectAndCompute(img1, noArray(), kpts1, desc1);
			}
			else //reload from previous iteration
			{
				kpts1 = kpts2;
				desc1 = desc2;
			}

			//calc second
			detector->detectAndCompute(img2, noArray(), kpts2, desc2);


			//Matchers
			//Flann
			desc1.convertTo(desc1, CV_32F); //need to convert to flann
			desc2.convertTo(desc2, CV_32F);
			FlannBasedMatcher matcher;

			//Brute Force
			//BFMatcher matcher(NORM_HAMMING);



			vector< vector<DMatch> > nn_matches; //all matches
			matcher.knnMatch(desc1, desc2, nn_matches, 2);


			vector<KeyPoint> matched1, matched2; //good matched points
			vector<DMatch> good_matches; //good matches
			std::vector<Point2f> obj; //image 1 points for rot matrix
			std::vector<Point2f> scene; //image 2 points for rot matrix
			vector<double> vectorintensity; //vector intensities for filtering


			vector<int> index(nn_matches.size(), 0); //index vector, holds the intensity order of matches

			for (int i = 0; i != index.size(); i++) index[i] = i; // fill index vector

			// Calculate intensities and min distance
			double mindist = 1000;

			for (int i = 0; i < nn_matches.size(); i++)
			{
				vectorintensity.push_back(sqrt(
					square(kpts1[nn_matches[i][0].queryIdx].pt.x - kpts2[nn_matches[i][0].trainIdx].pt.x) +
					square(kpts1[nn_matches[i][0].queryIdx].pt.y - kpts2[nn_matches[i][0].trainIdx].pt.y)
				)); //calculate intensities
				if (nn_matches[i][0].distance < mindist) mindist = nn_matches[i][0].distance; //get smallest distance value
			}

			//sort vectors by intensity from smallest
			sort(index.begin(), index.end(), //sort through intensity ascending order
				[&](const int& a, const int& b) {
				return (vectorintensity[a] < vectorintensity[b]);
			}
			);

			//Filter on distance (matching parameter)/vector intensity

			int iii = floor(index.size())*0.1; //discard first 10% 
											   //mindist taken out as even with 20*mindist it was too strong, only few points passed with akaze
			int i = iii;
			//Filter on vectors under 3*intensity of the lower 10%
			while (vectorintensity[index[i]] <= vectorintensity[index[iii]] * 3 && i < index.size() - 1)
			{
				//if (nn_matches[index[i]][0].distance<=20*mindist) //3 filter on distance
				//	{
				good_matches.push_back(nn_matches[index[i]][0]); //save out good matches
				matched1.push_back(kpts1[nn_matches[index[i]][0].queryIdx]); //save out coordinates
				matched2.push_back(kpts2[nn_matches[index[i]][0].trainIdx]);
				//-- Get the keypoints from the good matches
				//Same as before two lines, this will be used only for rotation matrix
				obj.push_back(kpts1[nn_matches[index[i]][0].queryIdx].pt);
				scene.push_back(kpts2[nn_matches[index[i]][0].trainIdx].pt);
				//}
				i++;
			}





			//rot matrix
			if (obj.size() > 10 && scene.size() > 10) //if points too low, don't try
			{
				//Homography part
				Mat cameraMatrix = (Mat1d(3, 3) << 700, 0, 240, 0, 700, 320, 0, 0, 1);
				Mat h = findHomography(obj, scene, CV_RANSAC,3,noArray(),4000,0.99999);
				vector<Mat> rotations, translations, normals;
				Mat r1;
				decomposeHomographyMat(h, cameraMatrix, rotations, translations, normals);

				vector<vector<double>> rotate; //Mat to double matrix

				r1 = rotations[0];
				rotate = { { r1.at<double>(0,0) ,r1.at<double>(1,0) ,r1.at<double>(2,0) }, { r1.at<double>(1,0) ,r1.at<double>(1,1) ,r1.at<double>(1,2) }, { r1.at<double>(2,0) ,r1.at<double>(2,1) ,r1.at<double>(2,2) } };
				theta = atan2(rotate[1][0], rotate[0][0])*a;
				rotb = rotb + theta;

			}




			//Calculations

			//for each vector on frame
			vector<translation> inframetransvector; //translation vectors inside of frame

			int number_of_features;
			number_of_features = min(4000, good_matches.size()); //this defines how many points will be used


// Intra frame
//loop through all the features in the frame, calculate, summarize

			dummytranslation = { current_frame,0,0,0,0 }; //dummy will sum up values
			dummyrotation = { current_frame,0,0 };

			for (int i = 0; i < number_of_features; i++)
			{
				dummytranslation.frame = i; //feature no.
				dummytranslation.x = dummytranslation.x + matched2[i].pt.x - matched1[i].pt.x; //sum of translation in frame
				dummytranslation.y = dummytranslation.y + matched2[i].pt.y - matched1[i].pt.y; //end-start
				dummytranslation.angle = 0; //will be calculated later
				dummytranslation.intensity = 0; //will be calculated later

				inframetransvector.push_back(dummytranslation); //write out to vector

				//red arrows showing all vectors, elongated 4 times
				arrowedLine(img3, matched1[i].pt,
					Point(matched1[i].pt + 4 * (matched2[i].pt - matched1[i].pt)),
					Scalar(0, 0, 255), 2, CV_AA);
			}


			//Normalize, calculate/add missing values

			//trans
			dummytranslation.frame = current_frame;
			int count = number_of_features;
			if (number_of_features == 0) count = 1; //don't divide with zero
			dummytranslation.x = dummytranslation.x / count; //normalize
			dummytranslation.y = dummytranslation.y / count;
			dummytranslation.angle = atan2((double)dummytranslation.y, (double)dummytranslation.x);//calculated from final translation points
			dummytranslation.intensity = sqrt(square(dummytranslation.y) + square(dummytranslation.x)); //calculated from final translation points
			transvector.push_back(dummytranslation);

			//rot
			dummyrotation.frame = current_frame;
			dummyrotation.angle = theta;
			//overallrot	
			dummyrotation.overallangle = rotb;

			rotvector.push_back(dummyrotation);

			//displacement
			dummydisplacement.frame = current_frame;
			dummydisplacement.x = cos(rotb / a + transvector.back().angle)*transvector.back().intensity; //relative displacement
			dummydisplacement.y = sin(rotb / a + transvector.back().angle)*transvector.back().intensity;
			if (current_frame == start) { //first frame different
				dummydisplacement.sumx = dummydisplacement.x;
				dummydisplacement.sumy = dummydisplacement.y;
			}
			else
			{
				dummydisplacement.sumx = dummydisplacement.x + dispvector.back().sumx;
				dummydisplacement.sumy = dummydisplacement.y + dispvector.back().sumy;
			}
			dummydisplacement.cmx = dummydisplacement.sumx*pixelCm;
			dummydisplacement.cmy = dummydisplacement.sumy*pixelCm;
			dispvector.push_back(dummydisplacement); //write out to vector
			//visualize dislocation with one green arrow from origo
			//elongate with factor 4
			arrowedLine(img3, centerpoint, Point(centerpoint.x + 4 * transvector.back().x, centerpoint.y + 4 * transvector.back().y), Scalar(0, 255, 0), 2, CV_AA);

			//visualize displacement on graph
			pointtomove = origo + Point(dummydisplacement.cmx*0.5, -dummydisplacement.cmy*0.5);
			line(image, pointfrom, pointtomove, color, 3, 8, 0);
			pointfrom = pointtomove;
			}

			// Output
			imshow("Displacement", image);  //draw frame
			imshow("Flowfield", img3);  //draw frame
			cout << current_frame << endl; //show current frame number

					
			//Write out
			f.open("xy.txt");//
			//for (int ii = 0; ii < thetavector.size(); ii++)
			for (int ii = 0; ii < dispvector.size(); ii++)
			{
				f.setf(std::ios::fixed, std::ios::floatfield); // floatfield set to fixed

				f << "Frame  " << setw(4) << transvector[ii].frame
					<< " Trans vector [" << setw(2) << transvector[ii].x <<"," << setw(2) << transvector[ii].y << "] "
					<< "   Relative rot angle " <<  setw(9) << rotvector[ii].angle << "   Overall rot angle " << setw(9) << rotvector[ii].overallangle
					<< "   Overall displacement [" <<  setprecision(2) << setw(4) << dispvector[ii].sumx << "," << setw(4) <<setprecision(2)  << dispvector[ii].sumy <<"]  ["<< dispvector[ii].cmx<<","<< dispvector[ii].cmy<<"] cm" <<endl
				 << endl;
			}
			f.close();


			int key_pressed=waitKey(1); 

			if (key_pressed == 'b' || key_pressed == 'B') //stops at frame
			{
				current_frame--;
				waitKey(0);
			}
			else				current_frame =current_frame+number_of_steps;
			
/* Don't run past the front/end of the AVI. */
	if (current_frame < 0) current_frame = 0;
	if (current_frame >= number_of_frames - number_of_steps) waitKey(0); //break;

	}

		}


//============================================================================
// Name        : Dip2.cpp
// Author      : Ronny Haensch
// Version     : 2.0
// Copyright   : -
// Description : 
//============================================================================

#include "Dip2.h"

// convolution in spatial domain
/*
src:     input image
kernel:  filter kernel
return:  convolution result
*/
Mat Dip2::spatialConvolution(Mat& src, Mat& kernel){
   
	// initialization	
	Mat  image, result;
	cv::cvtColor(src, image, CV_BGR2GRAY);
	image.convertTo(image, CV_32FC1);
	Mat conv = Mat::zeros(image.rows, image.cols, CV_32FC1);
	
	// define old and new filter Kernels																	  
	Mat  old_kernel = Mat::ones(3, 3, CV_32FC1);
	Mat  new_kernel = Mat::ones(old_kernel.rows, old_kernel.cols, CV_32FC1);

	// flip filter kernel
	for (int m = 0; m < old_kernel.rows; m++)
	{
		for (int n = 0; n < old_kernel.cols; n++)
		{
			new_kernel.at<float>(m, n) = old_kernel.at<float>(old_kernel.rows - 1 - m, old_kernel.cols - 1 - n);
		}
	}

	// make border with zero-padding from original_image 
	Mat  newimage = Mat::zeros(image.rows + 2, image.cols + 2, CV_32FC1);
	image.copyTo(newimage(Rect(1, 1, image.cols, image.rows)));

	// spatial convolution by a 3x3 kernel
	for (int i = 0; i <= newimage.rows - 3; i++)  
	{
		for (int j = 0; j <= newimage.cols - 3; j++)
		{
			Mat Roi_image = newimage(Rect(j, i, new_kernel.cols, new_kernel.rows));
			multiply(Roi_image, new_kernel, result);         														  
			conv.at<float>(i, j) = sum(result)[0];
		}
	}	
	return conv;
}

// the average filter
// HINT: you might want to use Dip2::spatialConvolution(...) within this function
/*
src:     input image
kSize:   window size used by local average
return:  filtered image
*/
Mat Dip2::averageFilter(Mat& src, int kSize){
  	
	Mat img    = src.clone();
	Mat average_image(img.size(), img.type());
	Mat kernel = Mat::ones(kSize, kSize, CV_32FC1) / (double)(kSize * kSize);
	Mat average_image = spatialConvolution(img, kernel);	

	return average_image;
}

// the median filter
/*
src:     input image
kSize:   window size used by median operation
return:  filtered image
*/
Mat Dip2::medianFilter(Mat& src, int kSize){
  
	// initialization	
	Mat dst = src.clone();
	Mat median_image(src.size(), src.type());

	// Make a border
	int added_rows = kSize / 2;
	int added_cols = kSize / 2;
	copyMakeBorder(src, dst, added_rows, added_rows, added_cols, added_cols, BORDER_CONSTANT, 0);
		
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {

			Mat Roi_image = dst(Rect(j, i, kSize, kSize)).clone();
	
			// initialize container 
			vector<float> Vector_ROI;

			// fill container with the ROI matrix elements 
			for (auto it = Roi_image.begin<float>(); it != Roi_image.end<float>(); it++)
				Vector_ROI.push_back(*it);

			// sort elements of the vector of ROI matrix elements
			std::sort(Vector_ROI.begin(), Vector_ROI.end());

			// create image
			median_image.at<float>(i, j) = Vector_ROI.at(Vector_ROI.size() / 2);
		}
	}
   return median_image;
}

// the bilateral filter
/*
src:     input image
kSize:   size of the kernel --> used to compute std-dev of spatial kernel
sigma:   standard-deviation of the radiometric kernel
return:  filtered image
*/
Mat Dip2::bilateralFilter(Mat& src, int kSize, double sigma){

	Mat dst = src.clone();
	int added_rows = kSize / 2;
	int added_cols = kSize / 2;
	copyMakeBorder(src, dst, added_rows, added_rows, added_cols, added_cols, BORDER_CONSTANT, 0);

	int mu = (kSize - 1) / 2;
	double sigma_spatial = 0.3*((kSize - 1)*0.5 - 1) + 0.8;

	// spatial kernel
	Mat spatialKernel(kSize, kSize, CV_32F);

	for (int i = 0; i < kSize; i++) {
		for (int j = 0; j < kSize; j++) {
			spatialKernel.at<float>(j, i) = exp(-((i - mu)*(i - mu) + (j - mu)*(j - mu)) / (2 * sigma_spatial*sigma_spatial));
		}
	}
	double weight = sum(spatialKernel)[0];
	spatialKernel /= weight;


	Point2i anchor(kSize / 2, kSize / 2);
	double k0 = 1 / (2 * sigma*sigma);
	Mat result(src.size(), src.type());
	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) {
			Mat subMat = dst(Rect(x, y, kSize, kSize)).clone();

			// radiometric kernel
			Mat radiometricKernel(kSize, kSize, CV_32F);
			for (int i = 0; i < kSize; i++) {
				for (int j = 0; j < kSize; j++) {
					double dVal = subMat.at<float>(anchor) - subMat.at<float>(j, i);
					radiometricKernel.at<float>(j, i) = exp(-(dVal*dVal) * k0);
					weight += radiometricKernel.at<float>(j, i);
				}
			}
			weight = sum(radiometricKernel)[0];
			radiometricKernel /= weight;

			Mat combinedKernel = spatialKernel.mul(radiometricKernel);
			combinedKernel /= sum(combinedKernel)[0];
	
			result.at<float>(y, x) = sum(subMat.mul(combinedKernel))[0];
			
		}
	}

	return result;

}

// the non-local means filter
/*
src:   		input image
searchSize: size of search region
sigma: 		Optional parameter for weighting function
return:  	filtered image
*/
Mat Dip2::nlmFilter(Mat& src, int searchSize, double sigma){
  
    return src.clone();

}

/* *****************************
  GIVEN FUNCTIONS
***************************** */

// function loads input image, calls processing function, and saves result
void Dip2::run(void){

	// window names
	string win1 = string("Original image");
	string win2 = string("noiseType_1.jpg");
	string win3 = string("noiseType_2.jpg");
	string win4 = string("restorated1.jpg");
	string win5 = string("restorated2.jpg");


   // load images as grayscale
	cout << "load images" << endl;
	Mat noise1 = imread("noiseType_1.jpg", 0);

	// show noisy image1
	namedWindow(win2.c_str());
	imshow(win2.c_str(), noise1);
	waitKey(0);

   if (!noise1.data){
	   cerr << "noiseType_1.jpg not found" << endl;
      cout << "Press enter to exit"  << endl;
      cin.get();
	   exit(-3);
	}
   noise1.convertTo(noise1, CV_32FC1);

	Mat noise2 = imread("noiseType_2.jpg",0);

	// show noisy image2
	namedWindow(win3.c_str());
	imshow(win3.c_str(), noise2);
	waitKey(0);

	if (!noise2.data){
	   cerr << "noiseType_2.jpg not found" << endl;
      cout << "Press enter to exit"  << endl;
      cin.get();
	   exit(-3);
	}
   noise2.convertTo(noise2, CV_32FC1);
   cout << "done" << endl;
	  
   // apply noise reduction
	// TO DO !!!
	// ==> Choose appropriate noise reduction technique with appropriate parameters
	// ==> "average" or "median"? Why?
	// ==> try also "bilateral" (and if implemented "nlm")
	cout << "reduce noise" << endl;
	Mat restorated1 = noiseReduction(noise1, "", 1);
	Mat restorated2 = noiseReduction(noise2, "", 1);
	cout << "done" << endl;
	  
	// save images
	cout << "save results" << endl;
	imwrite("restorated1.jpg", restorated1);
	// show restorated image1
	namedWindow(win4.c_str());
	imshow(win4.c_str(), noise2);
	waitKey(0);

	imwrite("restorated2.jpg", restorated2);
	// show restorated image2
	namedWindow(win5.c_str());
	imshow(win5.c_str(), noise2);
	waitKey(0);

	cout << "done" << endl;
	
	
}

// noise reduction
/*
src:     input image
method:  name of noise reduction method that shall be performed
	     "average" ==> moving average
         "median" ==> median filter
         "bilateral" ==> bilateral filter
         "nlm" ==> non-local means filter
kSize:   (spatial) kernel size
param:   if method == "bilateral", standard-deviation of radiometric kernel; if method == "nlm", (optional) parameter for similarity function
         can be ignored otherwise (default value = 0)
return:  output image
*/
Mat Dip2::noiseReduction(Mat& src, string method, int kSize, double param){

   // apply moving average filter
   if (method.compare("average") == 0){
      return averageFilter(src, kSize);
   }
   // apply median filter
   if (method.compare("median") == 0){
      return medianFilter(src, kSize);
   }
   // apply bilateral filter
   if (method.compare("bilateral") == 0){
      return bilateralFilter(src, kSize, param);
   }
   // apply adaptive average filter
   if (method.compare("nlm") == 0){
      return nlmFilter(src, kSize, param);
   }

   // if none of above, throw warning and return copy of original
   cout << "WARNING: Unknown filtering method! Returning original" << endl;
   cout << "Press enter to continue"  << endl;
   cin.get();
   return src.clone();

}

// generates and saves different noisy versions of input image
/*
fname:   path to the input image
*/
void Dip2::generateNoisyImages(string fname){
 
   // load image, force gray-scale
   cout << "load original image" << endl;
   Mat img = imread(fname, 0);
   if (!img.data){
      cerr << "ERROR: file " << fname << " not found" << endl;
      cout << "Press enter to exit"  << endl;
      cin.get();
      exit(-3);
   }

   // convert to floating point precision
   img.convertTo(img,CV_32FC1);
   cout << "done" << endl;

   // save original
   imwrite("original.jpg", img);
	  
   // generate images with different types of noise
   cout << "generate noisy images" << endl;

   // some temporary images
   Mat tmp1(img.rows, img.cols, CV_32FC1);
   Mat tmp2(img.rows, img.cols, CV_32FC1);
   // first noise operation
   float noiseLevel = 0.15;
   randu(tmp1, 0, 1);
   threshold(tmp1, tmp2, noiseLevel, 1, CV_THRESH_BINARY);
   multiply(tmp2,img,tmp2);
   threshold(tmp1, tmp1, 1-noiseLevel, 1, CV_THRESH_BINARY);
   tmp1 *= 255;
   tmp1 = tmp2 + tmp1;
   threshold(tmp1, tmp1, 255, 255, CV_THRESH_TRUNC);
   // save image
   imwrite("noiseType_1.jpg", tmp1);
    
   // second noise operation
   noiseLevel = 50;
   randn(tmp1, 0, noiseLevel);
   tmp1 = img + tmp1;
   threshold(tmp1,tmp1,255,255,CV_THRESH_TRUNC);
   threshold(tmp1,tmp1,0,0,CV_THRESH_TOZERO);
   // save image
   imwrite("noiseType_2.jpg", tmp1);

	cout << "done" << endl;
	cout << "Please run now: dip2 restorate" << endl;

}

// function calls some basic testing routines to test individual functions for correctness
void Dip2::test(void){

	test_spatialConvolution();
   test_averageFilter();
   test_medianFilter();

   cout << "Press enter to continue"  << endl;
   cin.get();

}

// checks basic properties of the convolution result
void Dip2::test_spatialConvolution(void){

   Mat input = Mat::ones(9,9, CV_32FC1);
   input.at<float>(4,4) = 255;
   Mat kernel = Mat(3,3, CV_32FC1, 1./9.);

   Mat output = spatialConvolution(input, kernel);
   
   if ( (input.cols != output.cols) || (input.rows != output.rows) ){
      cout << "ERROR: Dip2::spatialConvolution(): input.size != output.size --> Wrong border handling?" << endl;
      return;
   }
  if ( (sum(output.row(0) < 0).val[0] > 0) ||
           (sum(output.row(0) > 255).val[0] > 0) ||
           (sum(output.row(8) < 0).val[0] > 0) ||
           (sum(output.row(8) > 255).val[0] > 0) ||
           (sum(output.col(0) < 0).val[0] > 0) ||
           (sum(output.col(0) > 255).val[0] > 0) ||
           (sum(output.col(8) < 0).val[0] > 0) ||
           (sum(output.col(8) > 255).val[0] > 0) ){
         cout << "ERROR: Dip2::spatialConvolution(): Border of convolution result contains too large/small values --> Wrong border handling?" << endl;
         return;
   }else{
      if ( (sum(output < 0).val[0] > 0) ||
         (sum(output > 255).val[0] > 0) ){
            cout << "ERROR: Dip2::spatialConvolution(): Convolution result contains too large/small values!" << endl;
            return;
      }
   }
   float ref[9][9] = {{0, 0, 0, 0, 0, 0, 0, 0, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 1, 1, (8+255)/9., (8+255)/9., (8+255)/9., 1, 1, 0},
                      {0, 1, 1, (8+255)/9., (8+255)/9., (8+255)/9., 1, 1, 0},
                      {0, 1, 1, (8+255)/9., (8+255)/9., (8+255)/9., 1, 1, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 0, 0, 0, 0, 0, 0, 0, 0}};
   for(int y=1; y<8; y++){
      for(int x=1; x<8; x++){
         if (abs(output.at<float>(y,x) - ref[y][x]) > 0.0001){
            cout << "ERROR: Dip2::spatialConvolution(): Convolution result contains wrong values!" << endl;
            return;
         }
      }
   }
   input.setTo(0);
   input.at<float>(4,4) = 255;
   kernel.setTo(0);
   kernel.at<float>(0,0) = -1;
   output = spatialConvolution(input, kernel);
   if ( abs(output.at<float>(5,5) + 255.) < 0.0001 ){
      cout << "ERROR: Dip2::spatialConvolution(): Is filter kernel \"flipped\" during convolution? (Check lecture/exercise slides)" << endl;
      return;
   }
   if ( ( abs(output.at<float>(2,2) + 255.) < 0.0001 ) || ( abs(output.at<float>(4,4) + 255.) < 0.0001 ) ){
      cout << "ERROR: Dip2::spatialConvolution(): Is anchor point of convolution the centre of the filter kernel? (Check lecture/exercise slides)" << endl;
      return;
   }
   cout << "Message: Dip2::spatialConvolution() seems to be correct" << endl;
}

// checks basic properties of the filtering result
void Dip2::test_averageFilter(void){

   Mat input = Mat::ones(9,9, CV_32FC1);
   input.at<float>(4,4) = 255;

   Mat output = averageFilter(input, 3);
   
   if ( (input.cols != output.cols) || (input.rows != output.rows) ){
      cout << "ERROR: Dip2::averageFilter(): input.size != output.size --> Wrong border handling?" << endl;
      return;
   }
  if ( (sum(output.row(0) < 0).val[0] > 0) ||
           (sum(output.row(0) > 255).val[0] > 0) ||
           (sum(output.row(8) < 0).val[0] > 0) ||
           (sum(output.row(8) > 255).val[0] > 0) ||
           (sum(output.col(0) < 0).val[0] > 0) ||
           (sum(output.col(0) > 255).val[0] > 0) ||
           (sum(output.col(8) < 0).val[0] > 0) ||
           (sum(output.col(8) > 255).val[0] > 0) ){
         cout << "ERROR: Dip2::averageFilter(): Border of result contains too large/small values --> Wrong border handling?" << endl;
         return;
   }else{
      if ( (sum(output < 0).val[0] > 0) ||
         (sum(output > 255).val[0] > 0) ){
            cout << "ERROR: Dip2::averageFilter(): Result contains too large/small values!" << endl;
            return;
      }
   }
   float ref[9][9] = {{0, 0, 0, 0, 0, 0, 0, 0, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 1, 1, (8+255)/9., (8+255)/9., (8+255)/9., 1, 1, 0},
                      {0, 1, 1, (8+255)/9., (8+255)/9., (8+255)/9., 1, 1, 0},
                      {0, 1, 1, (8+255)/9., (8+255)/9., (8+255)/9., 1, 1, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 0, 0, 0, 0, 0, 0, 0, 0}};
   for(int y=1; y<8; y++){
      for(int x=1; x<8; x++){
         if (abs(output.at<float>(y,x) - ref[y][x]) > 0.0001){
            cout << "ERROR: Dip2::averageFilter(): Result contains wrong values!" << endl;
            return;
         }
      }
   }
   cout << "Message: Dip2::averageFilter() seems to be correct" << endl;
}

// checks basic properties of the filtering result
void Dip2::test_medianFilter(void){

   Mat input = Mat::ones(9,9, CV_32FC1);
   input.at<float>(4,4) = 255;

   Mat output = medianFilter(input, 3);
   
   if ( (input.cols != output.cols) || (input.rows != output.rows) ){
      cout << "ERROR: Dip2::medianFilter(): input.size != output.size --> Wrong border handling?" << endl;
      return;
   }
  if ( (sum(output.row(0) < 0).val[0] > 0) ||
           (sum(output.row(0) > 255).val[0] > 0) ||
           (sum(output.row(8) < 0).val[0] > 0) ||
           (sum(output.row(8) > 255).val[0] > 0) ||
           (sum(output.col(0) < 0).val[0] > 0) ||
           (sum(output.col(0) > 255).val[0] > 0) ||
           (sum(output.col(8) < 0).val[0] > 0) ||
           (sum(output.col(8) > 255).val[0] > 0) ){
         cout << "ERROR: Dip2::medianFilter(): Border of result contains too large/small values --> Wrong border handling?" << endl;
         return;
   }else{
      if ( (sum(output < 0).val[0] > 0) ||
         (sum(output > 255).val[0] > 0) ){
            cout << "ERROR: Dip2::medianFilter(): Result contains too large/small values!" << endl;
            return;
      }
   }
   for(int y=1; y<8; y++){
      for(int x=1; x<8; x++){
         if (abs(output.at<float>(y,x) - 1.) > 0.0001){
            cout << "ERROR: Dip2::medianFilter(): Result contains wrong values!" << endl;
            return;
         }
      }
   }
   cout << "Message: Dip2::medianFilter() seems to be correct" << endl;

}

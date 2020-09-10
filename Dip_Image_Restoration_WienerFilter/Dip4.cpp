//============================================================================
// Name        : Dip4.cpp
// Author      : Ronny Haensch
// Version     : 2.0
// Copyright   : -
// Description : 
//============================================================================

#include "Dip4.h"

// Performes a circular shift in (dx,dy) direction
/*
in       :  input matrix
dx       :  shift in x-direction
dy       :  shift in y-direction
return   :  circular shifted matrix
*/
Mat Dip4::circShift(Mat& in, int dx, int dy){
	Mat shifted(in.size(), in.type());
	for (int y = 0; y < in.rows; y++)
		for (int x = 0; x < in.cols; x++) {
			Point2i src(x - dx, y - dy);
			src.x = src.x < in.cols ? src.x : src.x - in.cols;
			src.y = src.y < in.rows ? src.y : src.y - in.rows;
			shifted.at<float>(y, x) = in.at<float>(src);
		}

	return shifted;  
}

// Function applies inverse filter to restorate a degraded image
/*
degraded :  degraded input image
filter   :  filter which caused degradation
return   :  restorated output image
*/
Mat Dip4::inverseFilter(Mat& degraded, Mat& filter){
	// Method1: Inverse filter computation from initial formula (with limitations)
	// resize kernel to image resolution + padding
	//Mat resizeKernel = Mat(degraded.rows, degraded.cols, degraded.type(), Scalar(0));
	//filter.copyTo(resizeKernel(Rect(0, 0, filter.cols, filter.rows)));

	//// shift resizekernel to the origin (use the circ shift function)
	//// the offset point (x, y) used for shifting here is the centre of the kernel
	//// Goal: to make the signal periodic
	//resizeKernel = circShift(resizeKernel, -filter.rows / 2, -filter.cols / 2);

	//// dft image
	//Mat imgDFT;
	//dft(degraded, imgDFT, 0); // "nonzeroRows" hint for faster processing

	//// dft resizekernel
	//Mat resizeKernelDft;
	//dft(resizeKernel, resizeKernelDft, 0);

	//// find inverse of the resizedftkernel
	//Mat inverseresizekernelDft = resizeKernelDft.inv();

	//// multiplicate both image and kernel spectrum in the frequency domain
	//Mat result;
	//mulSpectrums(imgDFT, inverseresizekernelDft, result, 0);

	//// inverse dft
	//dft(result, result, DFT_INVERSE + DFT_SCALE);

	//return result;

	// Method2: Replace the inverse filter by Qi = 1/T
	// resize kernel to image resolution + padding
	//Mat resizeKernel = Mat(degraded.rows, degraded.cols, degraded.type(), Scalar(0));
	//filter.copyTo(resizeKernel(Rect(0, 0, filter.cols, filter.rows)));

	//// shift resizekernel to the origin (use the circ shift function)
	//// the offset point (x, y) used for shifting here is the centre of the kernel
	//resizeKernel = circShift(resizeKernel, -filter.rows / 2, -filter.cols / 2);

	//// dft image
	//Mat imgDFT;
	//dft(degraded, imgDFT, 0); // "nonzeroRows" hint for faster processing

	//// dft resizekernel
	//Mat resizeKernelDft;
	//dft(resizeKernel, resizeKernelDft, 0);

	//// find max of the resizekerneldft
	//double min, max;
	//minMaxLoc(resizeKernelDft, &min, &max);
	//if ((min < 0) && (abs(min) > max))
	//	max = abs(min);
	//
	//// Choose Qi = 1/T where T = max of pixel value of resizekerneldft
	//Mat newresizeKernelDft = resizeKernelDft / max;

	//// multiplicate both image and kernel spectrum in the frequency domain
	//Mat result;
	//mulSpectrums(imgDFT, newresizeKernelDft, result, 0);

	//// inverse dft
	//dft(result, result, DFT_INVERSE + DFT_SCALE);

	//return result;
	
	// Method3: inverse filter with power spectrum values

	// resize kernel to image resolution + padding
	Mat resizeKernel = Mat(degraded.rows, degraded.cols, degraded.type(), Scalar(0));
	filter.copyTo(resizeKernel(Rect(0, 0, filter.cols, filter.rows)));

	// shift resizekernel to the origin (use the circ shift function)
	// the offset point (x, y) used for shifting here is the centre of the kernel
	resizeKernel = circShift(resizeKernel, -filter.rows / 2, -filter.cols / 2);

	// dft degraded image and kernel
	Mat degradedDFT, resizekernelDFT;
	dft(degraded, degradedDFT, DFT_COMPLEX_OUTPUT); 
	dft(resizeKernel, resizekernelDFT, DFT_COMPLEX_OUTPUT);

	// Split multi channel array into several single-channel arrays
	// G: Degraded, H:filter, F: estime of original image
	vector<Mat> S, P, O;
	split(degradedDFT, S);
	split(resizekernelDFT, P);

	// define the threshold of the filter spectrum
	const double epsilon = 0.1;
	double thres, max;

	
	// z = Re + i*Im => !z! = sqrt(Re^2 + Im^2)
	// P[0] = Re ;  P[1] = Im
	// Re^2 = Re*Re = multiply(P[0], P[0], ReRe); similar to Im*Im

	// matrix initialization 
	Mat ReRe, ImIm, abs, thres_abs;
    
	// compute the square of both the real and imaginary parts of 
	multiply(P[0], P[0], ReRe);
	multiply(P[1], P[1], ImIm);

	// compute the magnitude or amplitude or power of the spectrum: !z! = sqrt(Re^2 + Im^2)
	abs = ReRe + ImIm;
	sqrt(abs, abs);

	// find the maximum value of element of the filter spectrum matrix: P
	minMaxIdx(abs, (double*)0, &max);              // find the global maximum and minimum in a array

	//define threshold
	thres = epsilon * sqrt(max);

	// redefine the inverse filter matrix: called here "abs"
	// assumptions: Qi = 1/Pi if !pi! >= threshols
	// assumptions: Qi = 1/T if !pi!  < threshols
	for (int y = 0; y < abs.rows; y++) {
		for (int x = 0; x < abs.cols; x++) {
			if (abs.at<float>(y, x) < thres)
				P[0].at<float>(y, x) = thres;
		}
	}

												   // find spectrum power of the resizekerneldft: P(magnitude)
	Mat planes[] = { Mat_<double>(resizeKernel), Mat::zeros(resizeKernel.size(), CV_32FC1) };
	Mat complexI;
	merge(planes, 2, complexI);                    // Add to the expanded another plane with zeros

												   // dft resizekernel
	dft(complexI, complexI);                       // this way the result may fit in the source matrix

    // find magnitude of the filter spectrum
	split(complexI, planes);                       // planes[0] = Re(DFT(degraded), planes[1] = Im(DFT(degraded))
	magnitude(planes[0], planes[1], planes[0]);    // planes[0] = magnitude of P
	Mat magI = planes[0];


	// find max of the resizekerneldft
	double min, max;
	minMaxLoc(magI, &min, &max);
	
	// Choose Qi = 1/T where T = max of pixel value of resizekerneldft
	Mat newresizeKernelDft = complexI / max;

	// multiplicate both image and kernel spectrum in the frequency domain
	Mat result;
	mulSpectrums(imgDFT, newresizeKernelDft, result, 0);

	// inverse dft
	dft(result, result, DFT_INVERSE + DFT_SCALE);

	return result;
}

// Function applies wiener filter to restorate a degraded image
/*
degraded :  degraded input image
filter   :  filter which caused degradation
snr      :  signal to noise ratio of the input image
return   :   restorated output image
*/
Mat Dip4::wienerFilter(Mat& degraded, Mat& filter, double snr){
	// TO DO !!!
	// resize kernel to image resolution + padding
	//Mat resizeKernel = Mat(degraded.rows, degraded.cols, degraded.type(), Scalar(0));
	//filter.copyTo(resizeKernel(Rect(0, 0, filter.cols, filter.rows)));

	//// shift resizekernel to the origin (use the circ shift function)
	//// the offset point (x, y) used for shifting here is the centre of the kernel
	//resizeKernel = circShift(resizeKernel, -filter.rows / 2, -filter.cols / 2);

	//// dft image
	//Mat imgDFT;
	//dft(degraded, imgDFT, 0);                      // "nonzeroRows" hint for faster processing

	//// find spectrum power of the resizekerneldft: P(magnitude)
	//Mat planes[] = { Mat_<float>(resizeKernel), Mat::zeros(resizeKernel.size(), CV_32F) };
	//Mat complexI;
	//merge(planes, 2, complexI);                    // Add to the expanded another plane with zeros

	//// dft resizekernel
	//dft(complexI, complexI);                       // this way the result may fit in the source matrix

	//// find magnitude of P
	//split(complexI, planes);                       // planes[0] = Re(DFT(degraded), planes[1] = Im(DFT(degraded))
	//magnitude(planes[0], planes[1], planes[0]);    // planes[0] = magnitude of P
	//Mat magI = planes[0];

	//// find conjugate of P:

	//// compute the wiener filter: Qk = P(conjugate) / (mag^2 + 1/snr)
	//

	//// Choose Qi = 1/T where T = max of pixel value of resizekerneldft
 //   Mat newresizeKernelDft = resizeKernelDft / max;

	//// multiplicate both image and kernel spectrum in the frequency domain
	//Mat result;
	//mulSpectrums(imgDFT, newresizeKernelDft, result, 0);

	//// inverse dft
	//dft(result, result, DFT_INVERSE + DFT_SCALE);

	//return result;

	return degraded;
  
}

/* *****************************
  GIVEN FUNCTIONS
***************************** */

// function calls processing function
/*
in                   :  input image
restorationType     :  integer defining which restoration function is used
kernel               :  kernel used during restoration
snr                  :  signal-to-noise ratio (only used by wieder filter)
return               :  restorated image
*/
Mat Dip4::run(Mat& in, string restorationType, Mat& kernel, double snr){

   if (restorationType.compare("wiener")==0){
      return wienerFilter(in, kernel, snr);
   }else{
      return inverseFilter(in, kernel);
   }

}

// function degrades the given image with gaussian blur and additive gaussian noise
/*
img         :  input image
degradedImg :  degraded output image
filterDev   :  standard deviation of kernel for gaussian blur
snr         :  signal to noise ratio for additive gaussian noise
return      :  the used gaussian kernel
*/
Mat Dip4::degradeImage(Mat& img, Mat& degradedImg, double filterDev, double snr){

    int kSize = round(filterDev*3)*2 - 1;
   
    Mat gaussKernel = getGaussianKernel(kSize, filterDev, CV_32FC1);
    gaussKernel = gaussKernel * gaussKernel.t();

    Mat imgs = img.clone();
    dft( imgs, imgs, CV_DXT_FORWARD, img.rows);
    Mat kernels = Mat::zeros( img.rows, img.cols, CV_32FC1);
    int dx, dy; dx = dy = (kSize-1)/2.;
    for(int i=0; i<kSize; i++) for(int j=0; j<kSize; j++) kernels.at<float>((i - dy + img.rows) % img.rows,(j - dx + img.cols) % img.cols) = gaussKernel.at<float>(i,j);
	dft( kernels, kernels, CV_DXT_FORWARD );
	mulSpectrums( imgs, kernels, imgs, 0 );
	dft( imgs, degradedImg, CV_DXT_INV_SCALE, img.rows );
	
    Mat mean, stddev;
    meanStdDev(img, mean, stddev);

    Mat noise = Mat::zeros(img.rows, img.cols, CV_32FC1);
    randn(noise, 0, stddev.at<double>(0)/snr);
    degradedImg = degradedImg + noise;
    threshold(degradedImg, degradedImg, 255, 255, CV_THRESH_TRUNC);
    threshold(degradedImg, degradedImg, 0, 0, CV_THRESH_TOZERO);

    return gaussKernel;
}

// Function displays image (after proper normalization)
/*
win   :  Window name
img   :  Image that shall be displayed
cut   :  whether to cut or scale values outside of [0,255] range
*/
void Dip4::showImage(const char* win, Mat img, bool cut){

   Mat tmp = img.clone();

   if (tmp.channels() == 1){
      if (cut){
         threshold(tmp, tmp, 255, 255, CV_THRESH_TRUNC);
         threshold(tmp, tmp, 0, 0, CV_THRESH_TOZERO);
      }else
         normalize(tmp, tmp, 0, 255, CV_MINMAX);
         
      tmp.convertTo(tmp, CV_8UC1);
   }else{
      tmp.convertTo(tmp, CV_8UC3);
   }
   imshow(win, tmp);
}

// function calls basic testing routines to test individual functions for correctness
void Dip4::test(void){

   test_circShift();
   cout << "Press enter to continue"  << endl;
   cin.get();

}

void Dip4::test_circShift(void){
   
   Mat in = Mat::zeros(3,3,CV_32FC1);
   in.at<float>(0,0) = 1;
   in.at<float>(0,1) = 2;
   in.at<float>(1,0) = 3;
   in.at<float>(1,1) = 4;
   Mat ref = Mat::zeros(3,3,CV_32FC1);
   ref.at<float>(0,0) = 4;
   ref.at<float>(0,2) = 3;
   ref.at<float>(2,0) = 2;
   ref.at<float>(2,2) = 1;
   
   if (sum((circShift(in, -1, -1) == ref)).val[0]/255 != 9){
      cout << "ERROR: Dip4::circShift(): Result of circshift seems to be wrong!" << endl;
      return;
   }
   cout << "Message: Dip4::circShift() seems to be correct" << endl;
}

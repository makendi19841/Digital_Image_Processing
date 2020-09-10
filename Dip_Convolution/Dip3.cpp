//============================================================================
// Name    : Dip3.cpp
// Author   : Ronny Haensch
// Version    : 2.0
// Copyright   : -
// Description : 
//============================================================================

#include "Dip3.h"

// Generates a gaussian filter kernel of given size
/*
kSize:     kernel size (used to calculate standard deviation)
return:    the generated filter kernel
*/
Mat Dip3::createGaussianKernel(int kSize){
 
	// Initialization of Gaussian kernel parameters
	int mu = (kSize - 1) / 2;
	double sigma = (double)kSize / 5.0;
		
	// initialization of matrices
	Mat GaussianKernel(kSize, kSize, CV_32FC1);
    
	// calculation of the Gaussian kernel
	for (int i = 0; i < kSize; i++)
	{
		for (int j = 0; j < kSize; j++)
		{
			GaussianKernel.at<float>(i,j) = exp(-0.5 * ((i - mu)*(i - mu) + (j - mu)*(j - mu)) / (2 * CV_PI * sigma * sigma));
		}
	}
   
	// normalization gaussian kernel
	GaussianKernel /= sum(GaussianKernel)[0];

	return GaussianKernel;	
}


// Performes a circular shift in (dx,dy) direction
/*
in       input matrix
dx       shift in x-direction
dy       shift in y-direction
return   circular shifted matrix
*/

Mat Dip3::circShift(Mat& in, int dx, int dy) {
	Mat shifted(in.size(), in.type());
	for (int y = 0; y < in.rows; y++)
		for (int x = 0; x < in.cols; x++) {
			Point2i src(x - dx, y - dy);
			src.x = src.x < in.cols ? src.x : src.x - in.cols;
			src.y = src.y < in.rows ? src.y : src.y - in.rows;
			shifted.at<float>(y, x) = in.at<float>(src);
		}

	return shifted;

	/*Mat shifted(in.size(), in.type());
	for (int y = 0; y < in.rows; y++) {
		for (int x = 0; x < in.cols; x++) {
			Point2i src(y, x);
			src.y = (src.y + (in.rows - 1)) % in.rows;
			src.x = (src.x + (in.cols - 1)) % in.cols;
			shifted.at<float>(y, x) = in.at<float>(src);
		}
	}
	return shifted;*/
}

//Performes a convolution by multiplication in frequency domain
/*
in       input image
kernel   filter kernel
return   output image
*/
Mat Dip3::frequencyConvolution(Mat& in, Mat& kernel){

   // resize kernel to image resolution + padding
   Mat resizeKernel = Mat(in.rows, in.cols,  in.type(), Scalar(0));
   kernel.copyTo(resizeKernel(Rect(0, 0,  kernel.cols, kernel.rows)));

   // shift resizekernel to the origin (use the circ shift function)
   // the offset point (x, y) used for shifting here is the centre of the kernel
   resizeKernel = circShift(resizeKernel, -kernel.rows / 2, -kernel.cols / 2);

   // dft image
   Mat imgDFT;
   dft(in, imgDFT, 0); // "nonzeroRows" hint for faster processing
   
   // dft resizekernel
   Mat resizeKernelDft;
   dft(resizeKernel, resizeKernelDft, 0);

   // multiplicate both image and kernel spectrum in the frequency domain
   Mat result;
   mulSpectrums(imgDFT, resizeKernelDft, result, 0);
   
   // inverse dft
   dft(result, result, DFT_INVERSE + DFT_SCALE);

   return result;
}

// Performs UnSharp Masking to enhance fine image structures
/*
in       the input image
type     integer defining how convolution for smoothing operation is done
         0 <==> spatial domain; 1 <==> frequency domain; 2 <==> seperable filter; 3 <==> integral image
size     size of used smoothing kernel
thresh   minimal intensity difference to perform operation
scale    scaling of edge enhancement
return   enhanced image
*/
Mat Dip3::usm(Mat& in, int type, int size, double thresh, double scale){
   // calculate edge enhancement

   // unsharp mask: it is an effective high-pass filter

   // some temporary images 
   Mat src = in.clone();
   Mat tmp(in.rows, in.cols, CV_32FC1);
   
   
   // 1: smooth original image
   //    save result in tmp for subsequent usage
   switch(type){
      case 0:
         tmp = mySmooth(in, size, 0);
         break;
      case 1:
         tmp = mySmooth(in, size, 1);
         break;
      case 2: 
	tmp = mySmooth(in, size, 2);
        break;
      case 3: 
	tmp = mySmooth(in, size, 3);
        break;
      default:
         GaussianBlur(in, tmp, Size(floor(size/2)*2+1, floor(size/2)*2+1), size/5., size/5.);
   }

   // apply dft on tmp to get smoothed image
   Mat smoothedImg = tmp;

   // image substraction (contrast) to detect presence of edges
   Mat substractImg;
   subtract(src, smoothedImg, substractImg);

   // increase the contrast using scale
   /*Mat HighsmoothedImg, Scaledifference;*/

   Mat Scaledifference = Mat::zeros(src.size(), CV_32FC1);
   Mat HighsmoothedImg = Mat::zeros(src.size(), CV_32FC1);

   for (int i = 0; i < src.rows; i++)
   {
	   for (int j = 0; j < src.cols; j++)
	   {
		   Scaledifference.at<float>(i, j) = substractImg.at<float>(i, j) * scale;
		   HighsmoothedImg.at<float>(i, j) = (substractImg.at<float>(i, j) > thresh) ? Scaledifference.at<float>(i, j) : src.at<float>(i, j);
	   }
   }

   // add the three matrices: smoothedImg, substractImg, HighsmoothedImg.
   Mat result;
   add(src, Scaledifference, result);

   return result;
}

// convolution in spatial domain
/*
src:    input image
kernel:  filter kernel
return:  convolution result
*/
Mat Dip3::spatialConvolution(Mat& src, Mat& kernel){

    // initialization of matrices
    //Mat  conv, image;
	Mat  conv;

	// image conversion
	Mat image = src.clone();

   	//cv::cvtColor(inputImage, image, CV_BGR2GRAY);
   	image.convertTo(image, CV_32FC1);

	// initialization of matrices
   	Mat result    = Mat::zeros(image.rows, image.cols, CV_32FC1);
   	Mat  newimage = Mat::zeros(image.rows+2, image.cols+2, CV_32FC1); 
    
	// zero-padding for original_image 
	image.copyTo(newimage(Rect(1, 1, image.cols, image.rows)));

	// flip kernel
   	Mat  old_kernel = kernel.clone();
   	Mat  new_kernel = Mat::ones(old_kernel.rows, old_kernel.cols, CV_32FC1);
   	for (int m = 0; m < old_kernel.rows; m++) 
   	{
   		for (int n = 0; n < old_kernel.cols; n++) 
   		{
   			new_kernel.at<float>(m,n) = old_kernel.at<float>(old_kernel.rows - 1 - m, old_kernel.cols - 1 - n);
   		}
   	}
   
   // convolution in spatial domain
   	for (int i = 0; i <= newimage.rows - new_kernel.rows; i++)  // define general approach to deal with border checking
   	{
   		for (int j = 0; j <= newimage.cols - new_kernel.cols; j++)
   		{
   			// define ROI_image
   			Mat Roi_image = newimage(Rect(j, i, new_kernel.cols, new_kernel.rows));

   			// convolution Computation
   			multiply(Roi_image, new_kernel, conv);   

   			// fill the output matrix with pixel values from convolution      
   			src.at<float>(i, j) = sum(conv)[0];
   		}
   	}
   return src;
}

// convolution in spatial domain by seperable filters
/*
src:    input image
size     size of filter kernel
return:  convolution result
*/
Mat Dip3::seperableFilter(Mat& src, int size){

   // optional
   return src;

}

// convolution in spatial domain by integral images
/*
src:    input image
size     size of filter kernel
return:  convolution result
*/
Mat Dip3::satFilter(Mat& src, int size){

   // option
   return src;

}

/******************************
  GIVEN FUNCTIONS
******************************

// function calls processing function
/*
in       input image
type     integer defining how convolution for smoothing operation is done
         0 <==> spatial domain; 1 <==> frequency domain
size     size of used smoothing kernel
thresh   minimal intensity difference to perform operation
scale    scaling of edge enhancement
return   enhanced image
*/
Mat Dip3::run(Mat& in, int smoothType, int size, double thresh, double scale){

   return usm(in, smoothType, size, thresh, scale);

}


// Performes smoothing operation by convolution
/*
in       input image
size     size of filter kernel
type     how is smoothing performed?
return   smoothed image
*/
Mat Dip3::mySmooth(Mat& in, int size, int type){

   // create filter kernel
   Mat kernel = createGaussianKernel(size);
 
   // perform convolution
   switch(type){
     case 0: return spatialConvolution(in, kernel);	// 2D spatial convolution
     case 1: return frequencyConvolution(in, kernel);	// 2D convolution via multiplication in frequency domain
     case 2: return seperableFilter(in, size);	// seperable filter
     case 3: return satFilter(in, size);		// integral image
     default: return frequencyConvolution(in, kernel);
   }
}

// function calls basic testing routines to test individual functions for correctness
void Dip3::test(void){

   test_createGaussianKernel();
   test_circShift();
   test_frequencyConvolution();
   cout << "Press enter to continue"  << endl;
   cin.get();

}

void Dip3::test_createGaussianKernel(void){

   Mat k = createGaussianKernel(11);
   
   if ( abs(sum(k).val[0] - 1) > 0.0001){
      cout << "ERROR: Dip3::createGaussianKernel(): Sum of all kernel elements is not one!" << endl;
      return;
   }
   if (sum(k >= k.at<float>(5,5)).val[0]/255 != 1){
      cout << "ERROR: Dip3::createGaussianKernel(): Seems like kernel is not centered!" << endl;
      return;
   }
   cout << "Message: Dip3::createGaussianKernel() seems to be correct" << endl;
}

void Dip3::test_circShift(void){
   
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
      cout << "ERROR: Dip3::circShift(): Result of circshift seems to be wrong!" << endl;
      return;
   }
   cout << "Message: Dip3::circShift() seems to be correct" << endl;
}

void Dip3::test_frequencyConvolution(void){
   
   Mat input = Mat::ones(9,9, CV_32FC1);
   input.at<float>(4,4) = 255;
   Mat kernel = Mat(3,3, CV_32FC1, 1./9.);

   Mat output = frequencyConvolution(input, kernel);
   
   if ( (sum(output < 0).val[0] > 0) || (sum(output > 255).val[0] > 0) ){
      cout << "ERROR: Dip3::frequencyConvolution(): Convolution result contains too large/small values!" << endl;
      return;
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
            cout << "ERROR: Dip3::frequencyConvolution(): Convolution result contains wrong values!" << endl;
            return;
         }
      }
   }
   cout << "Message: Dip3::frequencyConvolution() seems to be correct" << endl;
}

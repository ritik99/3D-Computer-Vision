# 3D-Computer-Vision
This assignment aims to detect zero crossings and generate an image obtained after applying a difference of Gaussian filter. All necessary filters and convolution operations were written using Numpy.

The program has been written in Python 3.7. The libraries being used are:
1. OpenCV3
2. Numpy
3. Matplotlib
4. Math
5. mpl_toolkits

All the output images have been included in this folder. 

By default, the program will output the zero crossing image for 'butterfly.png'. Relevant codes for different parts of the questions have been included as comments. Please uncomment the required output.

The program has the following methods:
1. gau(x, y, sigma) - Returns the output of applying the Gaussian function for a particular co-ordinate.
Its inputs are the (x,y) position in the matrix, and the sigma of the Gaussian function

2. gaussianKernel(size, sigma) - Returns the normalised Gaussian filter of specified size. Its inputs are the size
of the filter and sigma of the Gaussian function

3. diffoGaussian(size, sigma1, sigma2) - Returns the Difference of Gaussian filter. Its inputs are:
a. size: Size of the kernel
b. sigma1: Sigma for one of the Gaussians
c. sigma2: Sigma for the other Gaussian

4. convolve(X, M) - This function performs the convolution operation. Its inputs are the image (in the form of
matrix) on which convolution has to be performed, and the filter which is used to perform convolution.

5. detectEdge(X, threshold) - This function detects the zero crossings in the convolved image. Its input is
the input image. The zero crossing uses four neighbours of the pixel to evaluate. If a pixel value is negative and one of its neighbours is positive the value of the pixel is set to 1 (since the pixel values changes from negative to positive, a zero crossing has occurred).
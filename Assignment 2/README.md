This assignment involved implementing SIFT (Scale Invariant Feature Transform) to match keypoints detected on two images of the same scene captured from different viewing angles.

The program has been written in Python 3.7. The libraries being used are:
1. OpenCV3
2. Numpy
3. Matplotlib
4. Math
5. mpl_toolkits
6. Random

All the output images have been included in this folder.

By default, the program will use the 'IMG0_001.jpg' and 'IMG1_002.jpg'. It will run the whole program by default which can take a long time. Following is a list of functions that have been implemented:

1. gau(x, y, sigma) - Returns the output of applying the Gaussian function for a particular co-ordinate.
Its inputs are the (x,y) position in the matrix, and the sigma of the Gaussian function

2. gaussianKernel(size, sigma) - Returns the normalised Gaussian filter of specified size. Its inputs are the size
of the filter and sigma of the Gaussian function

3. convolve(X, M) - This function performs the convolution operation. Its inputs are the image (in the form of
matrix) on which convolution has to be performed, and the filter which is used to perform convolution.

4. octave_gen(X, a, sigma, store_sigma) - This function generates an octave given the value of sigma, a and the original image X. It stores the values of the sigma being used in store_sigma.

5. dog_gen(octave) - This generates the difference of Gaussian scale space given an octave. Thus it makes 4 DoG images.

6. extremum(pixel_val, up, down, middle) - This function detects the extremas in the image. It receives the middle pixel value, and all the surrounding points in the same scale and the scale above and below it. It then compares the values with all the neighbouring pixels. If one of the neighbouring values is bigger than the point in consideration, then it cannot be a maxima. If one of the keypoints is lesser than the point in consideration, then it cannot be a minima. The function then returns false if it is neither a maxima not a minima, otherwise it returns true

7. key_point(mid_scale, up_scale, down_scale) - This part detects the key points, that is it finds the extremums in the DoG scale space image. It uses the above function (extremum) to find out the extremas in the image

8. get_submat(og_matrix, loc, size) - This is a helper function which returns a sub-matrix near a given location. It is used in descriptor() function to get 16 x 16 matrix around the given point



9. orientation(octave_img, blur_sigma, key_points) - This function gets the orientation of a given keypoint. It takes a list of key points. It calculates the theta and m values for the whole image, then around a key point, it makes a histogram with 36 bins. It also marks these keypoints in the image. Earlier, I was using cv.arrowedLine() to show the direction of orientation of the keypoint as well, but since the image size decreases, the arrow turns into boxed lines. Hence, a circle is drawn over each of the keypoints


10. descriptor(octave_img, blur_sigma, key_points) - This function outputs the descriptor for each of the keypoints. It calculates the m and theta values, and builds an 8-bin histogram for each of the 4x4 sub matrix in the 16 x 16 matrix surrounding a keypoint. The histograms are weighted by the m values which are smoothened using a Gaussian kernel of size 16 x 16 and sigma 8.


For faster usage, the images have already been generated and kept. The part which generates these images has been commented and is present in the code. (The results were found to be better and faster when cv.GaussianBlur() was used, hence this function was used to generate the images used)

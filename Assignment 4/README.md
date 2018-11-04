The program has been written using Python 3.7:
1. OpenCV3
2. Numpy
3. matplotlib
4. math
5. skimage.feature

The output image will be stored with the name 'outupt.jpg'.

The program first estimates the fundamental matrix. After that, the new location for each pixel is figured out by estimating its epipolar line in the new image and then finding the point on the line for which the distance of the descriptor is the closest. The descriptors are calculated using scikit-image's daisy descriptor. 

This is all accomplished using the following modules:
1. detectKeyPoints() - this returns the SIFT keypoints.
2. keyPointMatching() - This returns the fundamental matrix
3. compute_and_drawlines() - this is used to mark the keypoints on the image (not used)
4. find_mapping() - this is used to find the mapping of the old points to the new image plane
5. plot() - this is used to plot the image onto the new plane
6. lie_on_ine() - this is used to find the points that lie on a given epipolar line.

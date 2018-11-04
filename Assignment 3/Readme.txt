The program has been written in Python 3.7. The libraries being used are:
1. OpenCV3
2. Numpy
3. Matplotlib
4. Math
5. mpl_toolkits


All the output images have been included in this folder.

The program 'assignment3_1.py' corresponds to the first question and saves the panorama image with the name 'panorama.jpg'. The program titled 'assignment3_2.py' is for the second question and outputs the warped image with the name 'warped.jpg', where the first is obtained by warping points with different depths using its corresponding depth matrix, and the second is obtained using a common homography matrix for the whole image.

The homography matrix is estimated using RANSAC. First, the keypoints in both the reference and source images are determined using SIFT and then matched. Four points are then chosen at random to solver for the homography matrix using the Direct Linear Transform method. Next, the number of inliers and the number of outliers are calculated. A threshold is set for how many inliers is an acceptable number and is used to determine if the RANSAC algorithm can be stopped before its iterations complete or not. If the threshold is not satisfied for any of the homography matrices estimated, we calculate the final homography matrix using the best set of four points.

'assignment3_1.py' contains the following functions:
1. ransac - Computes the ransac algorithm and outputs the homography matrix.
2. calculateHomography - Returns homography matrix calculated using SVD.
3. inliers - Calculates the number of inliers for a given H matrix.
4. detectKeyPoints - Uses SIFT to detect keypoints in given image.
5. keyPointMatching - Used as helper function to match keypoints using FLANN matcher.


'assignment3_2.py' also contains these functions modified to solve the second question.
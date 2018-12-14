import cv2 as cv
import numpy as np
import sys
import matplotlib.pyplot as plt
#X = cv.resize(X, None, fx = 0.5, fy = 0.5, interpolation = cv.INTER_CUBIC)


'''This code defines the ransac algorithm. It takes as input the keypoints and the number of iterations to run. It 
calls the calculateHomography and inliers methods to generate the homography matrix and to generate the inliers. 
'''
def ransac(n_iterations, keyPoints):
    src_pts = keyPoints[0]
    dst_pts = keyPoints[1]
    #print(keyPoints1)
    inliers_list = []
    inliers_no = []
    for i in range(n_iterations):
        random_src_pts = []
        random_dst_pts = []

        for i in range(4):  #This part of the code randomly samples 4 keypoints which is used to generate homography matrix
            key = np.random.choice(len(dst_pts))
            random_src_pts.append(src_pts[key]) #Points to estimate homography
            random_dst_pts.append(dst_pts[key]) #Points to estimate homography

        k = len(random_src_pts)
        M = np.zeros((2 * k, 9))  #This part makes the equations used to calculate homography using DLT
        for i in range(0, 2 * k, 2):
            x, y = random_src_pts[int(i / 2)][0]
            x_, y_ = random_dst_pts[int(i / 2)][0]
            M[i + 0, :] = [x, y, 1, 0, 0, 0, -x * x_, -y * x_, -x_]
            M[i + 1, :] = [0, 0, 0, x, y, 1, -x * y_, -y * y_, -y_]

        H, s = calculateHomography(M, keyPoints, 2) # Returns the homography matrix in H, inliers in s
        inliers_list.append(s)
        inliers_no.append(len(s[0]))

        if len(s[0]) >= int(0.85 * len(dst_pts)):  #If the number of inliers is more than a set value, then that matrix is returned
            k = len(s[0])
            M = np.zeros((2 * k, 9))
            for i in range(0, 2 * k, 2):
                x, y = s[0][int(i / 2)][0]
                x_, y_ = s[1][int(i / 2)][0]
                M[i + 0, :] = [x, y, 1, 0, 0, 0, -x * x_, -y * x_, -x_]
                M[i + 1, :] = [0, 0, 0, x, y, 1, -x * y_, -y * y_, -y_]

            H, s = calculateHomography(M, keyPoints, 2) # Returns the homography matrix in H, inliers in s
            return (H / H[2, 2], s)

    inliers_no_max = np.argmax(np.array(inliers_no))    # Find out largest set of inlier points

    s = inliers_list[inliers_no_max]
    k = len(s[0])
    M = np.zeros((2 * k, 9))
    for i in range(0, 2 * k, 2):
        x, y = s[0][int(i / 2)][0]
        x_, y_ = s[1][int(i / 2)][0]
        M[i + 0, :] = [x, y, 1, 0, 0, 0, -x * x_, -y * x_, -x_]
        M[i + 1, :] = [0, 0, 0, x, y, 1, -x * y_, -y * y_, -y_]
    H, S = calculateHomography(M, keyPoints, 2) # Returns the homography matrix in H, inliers in s
    return (H / H[2, 2], s)

'''This method uses SVD to calculate the homography matrix'''

def calculateHomography(M, keyPoints, t):
    u, s, vh = np.linalg.svd(M, full_matrices=True) #SVD based on Direct Linear Transform method
    H = vh[-1].reshape(3, 3)
    srcPts = keyPoints[0]
    dstPts = keyPoints[1]
    S = inliers(srcPts, dstPts, H, t)
    return (H, S)


'''This method calculates the inliers and outliers by seeing if they lie within a specific distance or not'''
def inliers(srcPts, dstPts, H, t):
    temp = [[], []]
    for i in range(len(srcPts)):
        srcPt = srcPts[i]
        dstPt = dstPts[i]
        #print(srcPt)
        i, j = srcPt[0]
        vi = np.array([[i], [j], [1]])
        vf = np.matmul(H, vi)
        vf /= vf[2, 0]  # making the last coordinate 1

        # check if within some tolerance
        i, j = dstPt[0]
        vc = np.array([[i], [j], [1]])
        if np.linalg.norm(vf - vc) <= t:
            temp[0].append(srcPt)
            temp[1].append(dstPt)
    return temp

'''This method is used to extract and match keypoints using SIFT'''
def detectKeyPoints(imgA):
    sift = cv.xfeatures2d.SIFT_create()
    keyPoints, descriptors = sift.detectAndCompute(imgA, None)

    return (keyPoints, descriptors)


'''This method is used for keypoint matching. It also calls the ransac algorithm and returns the homography matrix'''

def keyPointMatching(imgA, imgB):


    keyPointsA, descriptorsA = detectKeyPoints(imgA)
    keyPointsB, descriptorsB = detectKeyPoints(imgB)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(descriptorsA, descriptorsB, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) > 10:
        src_pts = np.float32([keyPointsA[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keyPointsB[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        #M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        M, mask = ransac(1000, [src_pts, dst_pts])
        #print(M)
    else:
        print("Cannot find enough keypoints.")
    return M, mask

'''This method is used to warp an image given its initial and transformed co-ordinates'''
def warpImage(image, H):

    A_X = image.shape[0]
    A_Y = image.shape[1]
    max_X = 0
    max_Y = 0
    min_X = 10000
    mapping = {}
    for i in range(A_X):
        for j in range(A_Y):
            old_projective_coord = np.matrix([[j, i, 1]])
            new_projective_coord = H * np.transpose(old_projective_coord)
            new_projective_coord = new_projective_coord / new_projective_coord[2,0]

            if int(new_projective_coord[1, 0]) > max_X:
                max_X = int(new_projective_coord[0, 0])
            if int(new_projective_coord[0, 0]) > max_Y:
                max_Y = int(new_projective_coord[1, 0])
            if int(new_projective_coord[1, 0]) < min_X:
                min_X = int(new_projective_coord[0, 0])

            mapping[(i, j)] = (int(new_projective_coord[1, 0]), int(new_projective_coord[0, 0]))

    return (mapping, max_X, max_Y, min_X)
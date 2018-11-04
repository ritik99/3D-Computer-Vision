import cv2 as cv
import numpy as np
import sys
import matplotlib.pyplot as plt
#X = cv.resize(X, None, fx = 0.5, fy = 0.5, interpolation = cv.INTER_CUBIC)

path = './Images_Asgnmt3_1/'
'''This part of the code is used to load the images and to resize them'''

dataSetList = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6']

imagesNames = {'I1': ['STA_0031.JPG', 'STB_0032.JPG', 'STC_0033.JPG', 'STD_0034.JPG', 'STE_0035.JPG', 'STF_0036.JPG'],
               'I2': ['2_1.JPG', '2_2.JPG', '2_3.JPG', '2_4.JPG', '2_5.JPG'],
               'I3': ['3_1.JPG', '3_2.JPG', '3_3.JPG', '3_4.JPG', '3_5.JPG'],
               'I4': ['DSC02930.JPG', 'DSC02931.JPG', 'DSC02932.JPG', 'DSC02933.JPG', 'DSC02934.JPG'],
               'I5': ['DSC03002.JPG', 'DSC03003.JPG', 'DSC03004.JPG', 'DSC03005.JPG', 'DSC03006.JPG'],
               'I6': ['1_1.JPG', '1_2.JPG', '1_3.JPG', '1_4.JPG', '1_5.JPG']}

scale = {'I1': (0.3, 0.3), 'I2': (1, 1), 'I3': (1, 1), 'I4': (0.5, 0.5), 'I5': (0.5, 0.5), 'I6': (1, 1)}
'''Change the index of dataSetList (from 0 to 5) to load different datasets'''

dataSet = dataSetList[0]
images = []
finalimages = []


for i in range(5):
    print(path + dataSet + '/' + imagesNames[dataSet][i])
    temp = cv.imread(path + dataSet + '/' + imagesNames[dataSet][i])
    temp = cv.resize(temp, None, fx=scale[dataSet][0], fy=scale[dataSet][1], interpolation=cv.INTER_CUBIC)
    #tempGray = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)
    images.append(temp)
    finalimages.append(temp)


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
            random_src_pts.append(src_pts[key])
            random_dst_pts.append(dst_pts[key])

        k = len(random_src_pts)
        M = np.zeros((2 * k, 9))  #This part makes the equations used to calculate homography using DLT
        for i in range(0, 2 * k, 2):
            x, y = random_src_pts[int(i / 2)][0]
            x_, y_ = random_dst_pts[int(i / 2)][0]
            M[i + 0, :] = [x, y, 1, 0, 0, 0, -x * x_, -y * x_, -x_]
            M[i + 1, :] = [0, 0, 0, x, y, 1, -x * y_, -y * y_, -y_]

        H, s = calculateHomography(M, keyPoints, 2)
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

            H, s = calculateHomography(M, keyPoints, 2)
            return (H / H[2, 2], s)

    inliers_no_max = np.argmax(np.array(inliers_no))

    s = inliers_list[inliers_no_max]
    k = len(s[0])
    M = np.zeros((2 * k, 9))
    for i in range(0, 2 * k, 2):
        x, y = s[0][int(i / 2)][0]
        x_, y_ = s[1][int(i / 2)][0]
        M[i + 0, :] = [x, y, 1, 0, 0, 0, -x * x_, -y * x_, -x_]
        M[i + 1, :] = [0, 0, 0, x, y, 1, -x * y_, -y * y_, -y_]
    H, S = calculateHomography(M, keyPoints, 2)
    return (H / H[2, 2], s)

'''This method uses SVD to calculate the homography matrix'''

def calculateHomography(M, keyPoints, t):
    u, s, vh = np.linalg.svd(M, full_matrices=True)
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
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        #M, mask = ransac(1000, [src_pts, dst_pts])
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


for i in range(5):
    images[i] = cv.cvtColor(images[i], cv.COLOR_BGR2GRAY)

H_01, s_01 = keyPointMatching(images[1], images[0])
H_12, s_12 = keyPointMatching(images[2], images[1])
H_23, s_23 = keyPointMatching(images[3], images[2])
H_34, s_34 = keyPointMatching(images[4], images[3])

H_10 = np.linalg.inv(H_01)
H_21 = np.linalg.inv(H_12)
H_20 = np.dot(H_21, H_10)
H_24 = np.dot(H_23, H_34)

print(H_10)
print(H_21)
print(H_20)
print(H_24)

mapping_20, max_X_20, max_Y_20, min_X_20 = warpImage(images[0], H_20)
mapping_21, max_X_21, max_Y_21, min_X_21 = warpImage(images[1], H_21)
mapping_23, max_X_23, max_Y_23, min_X_23 = warpImage(images[3], H_23)
mapping_24, max_X_24, max_Y_24, min_X_24 = warpImage(images[4], H_24)

#('20 ', max_X_20, ' ', max_Y_20, ' ', min_X_20)
#print('21 ', max_X_21, ' ', max_Y_21, ' ', min_X_21)
#print('23 ', max_X_23, ' ', max_Y_23, ' ', min_X_23)
#print('24 ', max_X_24, ' ', max_Y_24, ' ', min_X_24)

'''This part of the code is used for mapping the pixels to their new co-ordinates and for interpolation'''
canvas = np.zeros((4000, 5000, 3), dtype=np.uint8)
if abs(min_X_20) < 20000:
    for i in range(images[0].shape[0]):
        for j in range(images[0].shape[1]):
            canvas[mapping_20[(i, j)][0] + 400, mapping_20[(i, j)][1] - 3050] = finalimages[0][i, j]
            canvas[mapping_20[(i, j)][0] + 400 + 1, mapping_20[(i, j)][1] - 3050] = finalimages[0][i, j]
            canvas[mapping_20[(i, j)][0] + 400, mapping_20[(i, j)][1] - 3050 + 1] = finalimages[0][i, j]
            canvas[mapping_20[(i, j)][0] + 400 + 1, mapping_20[(i, j)][1] - 3050 + 1] = finalimages[0][i, j]



for i in range(images[1].shape[0]):
    for j in range(images[1].shape[1]):
        canvas[mapping_21[(i, j)][0] + 400, mapping_21[(i, j)][1] - 3050] = finalimages[1][i, j]
        canvas[mapping_21[(i, j)][0] + 400 + 1, mapping_21[(i, j)][1] - 3050] = finalimages[1][i, j]
        canvas[mapping_21[(i, j)][0] + 400, mapping_21[(i, j)][1] - 3050 + 1] = finalimages[1][i, j]
        canvas[mapping_21[(i, j)][0] + 400 + 1, mapping_21[(i, j)][1] - 3050 + 1] = finalimages[1][i, j]

        canvas[mapping_21[(i, j)][0] + 400, mapping_21[(i, j)][1] - 3050] = finalimages[1][i, j]
        canvas[mapping_21[(i, j)][0] + 400 + 2, mapping_21[(i, j)][1] - 3050] = finalimages[1][i, j]
        canvas[mapping_21[(i, j)][0] + 400, mapping_21[(i, j)][1] - 3050 + 2] = finalimages[1][i, j]
        canvas[mapping_21[(i, j)][0] + 400 + 2, mapping_21[(i, j)][1] - 3050 + 2] = finalimages[1][i, j]

        canvas[mapping_21[(i, j)][0] + 400, mapping_21[(i, j)][1] - 3050] = finalimages[1][i, j]
        canvas[mapping_21[(i, j)][0] + 400 + 3, mapping_21[(i, j)][1] - 3050] = finalimages[1][i, j]
        canvas[mapping_21[(i, j)][0] + 400, mapping_21[(i, j)][1] - 3050 + 3] = finalimages[1][i, j]
        canvas[mapping_21[(i, j)][0] + 400 + 3, mapping_21[(i, j)][1] - 3050 + 3] = finalimages[1][i, j]

        canvas[mapping_21[(i, j)][0] + 400, mapping_21[(i, j)][1] - 3050] = finalimages[1][i, j]
        canvas[mapping_21[(i, j)][0] + 400 + 4, mapping_21[(i, j)][1] - 3050] = finalimages[1][i, j]
        canvas[mapping_21[(i, j)][0] + 400, mapping_21[(i, j)][1] - 3050 + 4] = finalimages[1][i, j]
        canvas[mapping_21[(i, j)][0] + 400 + 4, mapping_21[(i, j)][1] - 3050 + 4] = finalimages[1][i, j]

for i in range(images[2].shape[0]):
    for j in range(images[2].shape[1]):
        canvas[i + 400, j - 3050] = finalimages[2][i, j]

for i in range(images[3].shape[0]):
    for j in range(images[3].shape[1]):
        if j in list(range(1, 10)):
            canvas[mapping_23[(i, j)][0] + 400, mapping_23[(i, j)][1] - 3050] = finalimages[3][i, j]
            canvas[mapping_23[(i, j)][0] + 400 + 1, mapping_23[(i, j)][1] - 3050] = finalimages[3][i, j]
            canvas[mapping_23[(i, j)][0] + 400, mapping_23[(i, j)][1] - 3050 + 1] = finalimages[3][i, j]
            canvas[mapping_23[(i, j)][0] + 400 + 1, mapping_23[(i, j)][1] - 3050 + 1] = finalimages[3][i, j]
        else:
            canvas[mapping_23[(i, j)][0] + 400, mapping_23[(i, j)][1] - 3050] = finalimages[3][i, j]
            canvas[mapping_23[(i, j)][0] + 400 + 1, mapping_23[(i, j)][1] - 3050] = finalimages[3][i, j]
            canvas[mapping_23[(i, j)][0] + 400, mapping_23[(i, j)][1] - 3050 + 1] = finalimages[3][i, j]
            canvas[mapping_23[(i, j)][0] + 400 + 1, mapping_23[(i, j)][1] - 3050 + 1] = finalimages[3][i, j]


for i in range(images[4].shape[0]):
    for j in range(images[4].shape[1]):
        canvas[mapping_24[(i, j)][0] + 400, mapping_24[(i, j)][1] - 3050] = finalimages[4][i, j]
        canvas[mapping_24[(i, j)][0] + 400 + 1, mapping_24[(i, j)][1] - 3050] = finalimages[4][i, j]
        canvas[mapping_24[(i, j)][0] + 400, mapping_24[(i, j)][1] - 3050 + 1] = finalimages[4][i, j]
        canvas[mapping_24[(i, j)][0] + 400 + 1, mapping_24[(i, j)][1] - 3050 + 1] = finalimages[4][i, j]

cv.imwrite("panorama.jpg", canvas)


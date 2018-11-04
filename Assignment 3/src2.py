import cv2 as cv
import numpy as np
import sys
import matplotlib.pyplot as plt
#X = cv.resize(X, None, fx = 0.5, fy = 0.5, interpolation = cv.INTER_CUBIC)

path = './RGBD dataset/'

dataSetList = ['000000029', '000000292', '000000705', '000001524', '000002812', '000003345']

imagesNames = {'000000029_rgb': ['im_0.jpg', 'im_1.jpg', 'im_2.jpg', 'im_3.jpg'],
               '000000292_rgb': ['im_0.jpg', 'im_1.jpg', 'im_2.jpg', 'im_3.jpg'],
               '000000705_rgb': ['im_0.jpg', 'im_1.jpg', 'im_2.jpg', 'im_3.jpg'],
               '000001524_rgb': ['im_0.jpg', 'im_1.jpg', 'im_2.jpg', 'im_3.jpg'],
               '000002812_rgb': ['im_0.jpg', 'im_1.jpg', 'im_2.jpg', 'im_3.jpg'],
               '000003345_rgb': ['im_0.jpg', 'im_1.jpg', 'im_2.jpg', 'im_3.jpg'],
               '000000029_depth': ['depth_0.jpg', 'depth_1.jpg', 'depth_2.jpg', 'depth_3.jpg'],
               '000000292_depth': ['depth_0.jpg', 'depth_1.jpg', 'depth_2.jpg', 'depth_3.jpg'],
               '000000705_depth': ['depth_0.jpg', 'depth_1.jpg', 'depth_2.jpg', 'depth_3.jpg'],
               '000001524_depth': ['depth_0.jpg', 'depth_1.jpg', 'depth_2.jpg', 'depth_3.jpg'],
               '000002812_depth': ['depth_0.jpg', 'depth_1.jpg', 'depth_2.jpg', 'depth_3.jpg'],
               '000003345_depth': ['depth_0.jpg', 'depth_1.jpg', 'depth_2.jpg', 'depth_3.jpg']}

scale = {'000000029': (0.3, 0.3), '000000292': (1, 1), '000000705': (0.5, 0.5), '000001524': (0.5, 0.5), '000002812': (0.5, 0.5), '000003345': (0.5, 0.5)}
dataSet = dataSetList[2]
rgb_images = []
d_images = []

key_point_dmapping = {}


for i in range(4):
    rgb_image = cv.imread(path + dataSet + '/' + imagesNames[dataSet + str('_rgb')][i])
    d_image = cv.imread(path + dataSet + '/' + imagesNames[dataSet + str('_depth')][i])
    rgb_images.append(rgb_image)
    d_image = d_image.astype(int)
    d_images.append(d_image)


#print('here')

#print('INITIAL D VAL ', d_images[2][87, 0])
d_images[2][87, 0] = 260
#print(d_images[2][87, 0])
for i in range(len(d_images)):
    for j in range(d_images[i].shape[0]):
        for k in range(d_images[i].shape[1]):
            #print('before ', d_images[i][j, k])
            #if i == 2:
               # print('(', j , k, ')', ' ', ((d_images[i][j, k] // 25) * 25) + 10)
            d_images[i][j, k] = ((d_images[i][j, k] // 50) * 50) + 25
            #print('after ', d_images[i][j, k])
#print('INITIAL D VAL ', d_images[2][87, 0])
#print(d_images[2])

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


def detectKeyPoints(imgA):
    sift = cv.xfeatures2d.SIFT_create()
    keyPoints, descriptors = sift.detectAndCompute(imgA, None)

    return (keyPoints, descriptors)

def keyPointMatching(imgA, imgB, d_image, depth_val):


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
        src_pts = []
        dst_pts = []
        for m in good:
            if d_image[int(keyPointsA[m.queryIdx].pt[1]), int(keyPointsA[m.queryIdx].pt[0]), 0] == depth_val:
                try:
                    src_pts.append(np.float32(keyPointsA[m.queryIdx].pt))
                    dst_pts.append(np.float32(keyPointsB[m.trainIdx].pt))
                    #src_pts = np.float32([keyPointsA[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    #dst_pts = np.float32([keyPointsB[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                except:
                    print('Error ', m.trainIdx, ' ')
        #M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        src_pts = np.float32(src_pts).reshape(-1, 1, 2)
        dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)
        #src_pts = np.float32([keyPointsA[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        #dst_pts = np.float32([keyPointsB[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        if src_pts.size != 0 and dst_pts.size != 0:
            M, mask = ransac(1000, [src_pts, dst_pts])
        else:
            return (0, 0)
        print(M)
    else:
        return (0, 0)
    return M, mask

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


#print('here')
H = []
s = []
color_image = rgb_images[2]
depth_image = d_images[2]
dstcolor_image = rgb_images[1]
for i in range(6):
    tempH, temps = keyPointMatching(color_image, dstcolor_image, depth_image, (50 * i) + 25)
    H.append(tempH)
    s.append(temps)

for i in range(len(H)):
    try:
        if H[i] == 0:
            H[i] = H[i-1]
    except:
        print('')

canvas = np.zeros((6000, 8000, 3))
mapping = {}
for i in range(color_image.shape[0]):
    for j in range(color_image.shape[1]):
        if d_images[2][i, j, 0] == 25:
            old_projective_coord = np.matrix([[j, i, 1]])
            new_projective_coord = H[0] * np.transpose(old_projective_coord)
            new_projective_coord = new_projective_coord / new_projective_coord[2, 0]

            mapping[(i, j)] = (int(new_projective_coord[1, 0]), int(new_projective_coord[0, 0]))

        elif depth_image[i, j, 0] == 75:
            old_projective_coord = np.matrix([[j, i, 1]])
            new_projective_coord = H[1] * np.transpose(old_projective_coord)
            new_projective_coord = new_projective_coord / new_projective_coord[2, 0]

            mapping[(i, j)] = (int(new_projective_coord[1, 0]), int(new_projective_coord[0, 0]))

        elif depth_image[i, j, 0] == 125:
            old_projective_coord = np.matrix([[j, i, 1]])
            new_projective_coord = H[2] * np.transpose(old_projective_coord)
            new_projective_coord = new_projective_coord / new_projective_coord[2, 0]

            mapping[(i, j)] = (int(new_projective_coord[1, 0]), int(new_projective_coord[0, 0]))

        elif depth_image[i, j, 0] == 175:
            old_projective_coord = np.matrix([[j, i, 1]])
            new_projective_coord = H[3] * np.transpose(old_projective_coord)
            new_projective_coord = new_projective_coord / new_projective_coord[2, 0]

            mapping[(i, j)] = (int(new_projective_coord[1, 0]), int(new_projective_coord[0, 0]))

        elif depth_image[i, j, 0] == 225:
            old_projective_coord = np.matrix([[j, i, 1]])
            new_projective_coord = H[4] * np.transpose(old_projective_coord)
            new_projective_coord = new_projective_coord / new_projective_coord[2, 0]

            mapping[(i, j)] = (int(new_projective_coord[1, 0]), int(new_projective_coord[0, 0]))

        elif depth_image[i, j, 0] == 275:
            old_projective_coord = np.matrix([[j, i, 1]])
            new_projective_coord = H[5] * np.transpose(old_projective_coord)
            new_projective_coord = new_projective_coord / new_projective_coord[2, 0]

            mapping[(i, j)] = (int(new_projective_coord[1, 0]), int(new_projective_coord[0, 0]))

        '''elif depth_image[i, j, 0] == 160:
            old_projective_coord = np.matrix([[j, i, 1]])
            new_projective_coord = H[6] * np.transpose(old_projective_coord)
            new_projective_coord = new_projective_coord / new_projective_coord[2, 0]

            mapping[(i, j)] = (int(new_projective_coord[1, 0]), int(new_projective_coord[0, 0]))

        elif depth_image[i, j, 0] == 185:
            old_projective_coord = np.matrix([[j, i, 1]])
            new_projective_coord = H[7] * np.transpose(old_projective_coord)
            new_projective_coord = new_projective_coord / new_projective_coord[2, 0]

            mapping[(i, j)] = (int(new_projective_coord[1, 0]), int(new_projective_coord[0, 0]))

        elif depth_image[i, j, 0] == 210:
            old_projective_coord = np.matrix([[j, i, 1]])
            new_projective_coord = H[8] * np.transpose(old_projective_coord)
            new_projective_coord = new_projective_coord / new_projective_coord[2, 0]

            mapping[(i, j)] = (int(new_projective_coord[1, 0]), int(new_projective_coord[0, 0]))

        elif depth_image[i, j, 0] == 235:
            old_projective_coord = np.matrix([[j, i, 1]])
            new_projective_coord = H[9] * np.transpose(old_projective_coord)
            new_projective_coord = new_projective_coord / new_projective_coord[2, 0]

            mapping[(i, j)] = (int(new_projective_coord[1, 0]), int(new_projective_coord[0, 0]))

        elif depth_image[i, j, 0] == 260:
            old_projective_coord = np.matrix([[j, i, 1]])
            new_projective_coord = H * np.transpose(old_projective_coord)
            new_projective_coord = new_projective_coord / new_projective_coord[2, 0]

            mapping[(i, j)] = (int(new_projective_coord[1, 0]), int(new_projective_coord[0, 0]))'''
#print(mapping[(0, 0)])
#print(depth_image[0, 0])
for i in range(color_image.shape[0]):
    for j in range(color_image.shape[1]):
        canvas[mapping[(i, j)][0] + 400, mapping[(i, j)][1] - 1000] = color_image[i, j]
        canvas[mapping[(i, j)][0] + 400 + 1, mapping[(i, j)][1] - 1000] = color_image[i, j]
        canvas[mapping[(i, j)][0] + 400, mapping[(i, j)][1] + 1 - 1000] = color_image[i, j]
        canvas[mapping[(i, j)][0] + 400 + 1, mapping[(i, j)][1] + 1 - 1000] = color_image[i, j]
cv.imwrite("warped.jpg", canvas)
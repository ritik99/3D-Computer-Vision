import cv2 as cv
import numpy as np
import sys
import matplotlib.pyplot as plt

from helper_2 import *

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
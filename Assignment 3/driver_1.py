'''
Author: Ritik Dutta
'''
from helper_1 import *
import cv2 as cv
import numpy as np
import sys
import matplotlib.pyplot as plt

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
cv.imshow('image',canvas)
cv.waitKey(0)
cv.imwrite("panorama.jpg", canvas)
print('Done.')
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf, precision = 3)
from mpl_toolkits import mplot3d
import math
import random

one_sigmas = []
two_sigmas = []
three_sigmas = []
four_sigmas = []

# Implements the Gaussian function
def gau(x, y, sigma):
    return 1 / (2 * np.pi * (sigma) ** 2) * np.exp(
        -(x ** 2 + y ** 2) / (2 * (sigma) ** 2))


# This function outputs the Gaussian Kernel
def gaussianKernel(size, sigma):
    M = np.zeros((size, size), dtype=np.float32)
    s = 0
    if size % 2 != 0:
        for i in range(-(size // 2), size // 2 + 1):  # Calculates the value of each element of the kernel by calling gau()
            for j in range(-(size // 2), size // 2 + 1):
                M[i + size // 2, j + size // 2] = gau(i, j, sigma)
                s = s + M[i + size // 2, j + size // 2]
    else:
        for i in range(-(size // 2), size // 2):  # Calculates the value of each element of the kernel by calling gau()
            for j in range(-(size // 2), size // 2):
                M[i + size // 2, j + size // 2] = gau(i, j, sigma)
                s = s + M[i + size // 2, j + size // 2]

    for i in range(M.shape[0]):  # This part normalises the kernel
        for j in range(M.shape[1]):
            M[i, j] /= s
    M = M.astype(np.float32)
    #print(M)  # prints the kernel. Comment the line if not needed
    return M


# This function performs the convolution. It performs the padding operation as well.
def convolve(X, M):
    padding_v = M.shape[0] // 2
    tempX = X.shape[0]
    tempY = X.shape[1]
    X_copy = np.zeros((padding_v + tempX + padding_v, padding_v + tempY + padding_v))
    for i in range(padding_v, padding_v + tempX):  # This part performs the padding operation
        for j in range(padding_v, padding_v + tempY):
            X_copy.itemset((i, j), X[i - padding_v, j - padding_v])
    # print(X_copy)

    Y = np.zeros((tempX, tempY), dtype=np.float32)
    posX = 0
    posY = 0
    for i in range(0, X_copy.shape[0] - M.shape[0]):  # This part performs the actual convolution
        for j in range(0, X_copy.shape[1] - M.shape[1]):
            t = X_copy[np.ix_([k for k in range(i, M.shape[0] + i)], [l for l in range(j, M.shape[1] + j)])]
            tt = 0
            for k in range(M.shape[0]):
                for l in range(M.shape[1]):
                    tt = tt + M[k, l] * t[k, l]
            Y[posX, posY] = tt
            posY += 1
        posY = 0
        posX += 1
    return Y


def octave_gen(X, a, sigma, store_sigma):
    octave = []
    #print(a, sigma)
    for i in range(5):
        print('now ', i)
        temp = gaussianKernel(9, (a ** (i)) * sigma)
        store_sigma.append((a ** (i)) * sigma)
        octave.append(convolve(X, temp))

    return octave

def dog_gen(octave):
    gen = []
    for i in range(4):
        gen.append(octave[i + 1] - octave[i])
    return gen

#This function detects the extremums in the image. It receives the middle pixel value, and all the surrounding points
#in the same scale and the scale above and below it. It then compares the values with all the neighbouring pixels.
#If one of the neighbouring values is bigger than the point in consideration, then it cannot be a maxima. If one of the
# keypoints is lesser than the point in consideration, then it cannot be a minima. The function then returns false if
#it is neither a maxima not a minima, otherwise it returns true


def extremum(pixel_val, up, down, middle):
    flag = True
    maxima = True
    minima = True
    for i in range(up.shape[0]):
        for j in range(up.shape[1]):
            if pixel_val < up[i, j] or pixel_val < down[i, j] or pixel_val < middle[i, j]:
                maxima = False
                break
        if maxima == False:
            break

    for i in range(up.shape[0]):
        for j in range(up.shape[1]):
            if pixel_val > up[i, j] or pixel_val > down[i, j] or pixel_val > middle[i, j]:
                minima = False
                break
        if minima == False:
            break
    if maxima == False and minima == False:
        return False
    else:
        return True

############################## KEY-POINTS ###################################

#This part detects the key points, that is it finds the extremums in the DoG scale space image. It uses the above function
#(extremum) to find out the extremas in the image


def key_point(mid_scale, up_scale, down_scale):
    mid_scale = cv.cvtColor(mid_scale, cv.COLOR_BGR2GRAY)
    up_scale = cv.cvtColor(up_scale, cv.COLOR_BGR2GRAY)
    down_scale = cv.cvtColor(down_scale, cv.COLOR_BGR2GRAY)
    key_points = []
    print(mid_scale.shape)
    x, y = mid_scale.shape
    for i in range(1, x):
        for j in range(1, y):
            if extremum(mid_scale[i, j], up_scale[i-1:i+1, j-1:j+1], down_scale[i-1:i+1, j-1:j+1], mid_scale[i-1: i+1, j-1:j+1]):
                key_points.append((i, j))

    return key_points

#This is a helper function which returns a sub-matrix near a given location. It is used in descriptor() function to get
#16 x 16 matrix around the given point


def get_submat(og_matrix, loc, size):
    temp = np.zeros((size, size))
    x = loc[0]
    y = loc[1]

    for i in range(size):
        for j in range(size):
            try:
                temp[i,j] = og_matrix[x + i, y + j]
            except:
                temp[i, j] = -1

    return temp


############################## ORIENTATION ###########################################

#This function gets the orientation of a given keypoint. It takes a list of key points. It calculates the theta and m
# values for the whole image, then around a key point, it makes a histogram with 36 bins. It also marks these keypoints in
# the image. Earlier, I was using cv.arrowedLine() to show the direction of orientation of the keypoint as well, but since
# the image size decreases, the arrow turns into boxed lines. Hence, a circle is drawn over each of the keypoints

def orientation(octave_img, blur_sigma, key_points):
    temp_scale = cv.copyMakeBorder(octave_img, 9, 8, 9, 8, cv.BORDER_REFLECT, None, 1)
    temp_m = np.zeros((octave_img.shape[0] + 15, octave_img.shape[1] + 15))
    temp_theta = np.zeros((octave_img.shape[0] + 15, octave_img.shape[1] + 15))
    for i in range(1, octave_img.shape[0] + 16):
        for j in range(1, octave_img.shape[1] + 16):
            temp_m[i - 1, j - 1] = ((temp_scale[i + 1, j] - temp_scale[i - 1, j])**2 + (temp_scale[i, j + 1] - temp_scale[i, j - 1])**2 )**0.5
            temp_theta[i - 1, j - 1] = np.degrees(math.atan2((temp_scale[i, j + 1] - temp_scale[i, j - 1]), (temp_scale[i + 1, j] - temp_scale[i - 1, j]))) + 180
            print(temp_theta[i-1, j-1])

    gaussian_ker = gaussianKernel(16, 1.5 * blur_sigma)
    print(temp_theta)
    bins_of_bin = {}
    for loc in key_points:
        x = loc[0]
        y = loc[1]

        bins = {}

        for i in range(16):
            for j in range(16):
                try:
                    #print((temp_theta[x + i - 8, y + j - 8] - (temp_theta[x + i - 8, y + j - 8]%10)))
                    bins[int((temp_theta[x + i - 8, y + j - 8] - (temp_theta[x + i - 8, y + j - 8]%10))/10)] += gaussian_ker[i, j] * temp_m[x + i - 8, y + j - 8]
                except:
                    bins[int((temp_theta[x + i - 8, y + j - 8] - (temp_theta[x + i - 8, y + j - 8]%10))/10)] = gaussian_ker[i, j] * temp_m[x + i - 8, y + j - 8]

        bins_of_bin[loc] = bins
    print(len(bins_of_bin))




    for loc in key_points:
        print('looping ', loc)
        max_bin = max(bins_of_bin[loc], key=lambda key: bins_of_bin[loc][key])
        x1 = loc[0]
        y1 = loc[1]
        x2 = int(x1 + bins_of_bin[loc][max_bin] * math.cos(math.radians(max_bin + 5)))
        y2 = int(y1 + bins_of_bin[loc][max_bin] * math.sin(math.radians(max_bin + 5)))
        #cv.arrowedLine(octave_img, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv.circle(octave_img, (x1, y1), 1, (0, 255, 0), -1)

    plt.imshow(octave_img, cmap = 'gray')
    plt.show()



####################### DESCRIPTOR #######################################

#This function outputs the descriptor for each of the keypoints. It calculates the m and theta values, and builds
# an 8-bin histogram for each of the 4x4 sub matrix in the 16 x 16 matrix surrounding a keypoint. The histograms are
# weighted by the m values which are smoothened using a Gaussian kernel of size 16 x 16 and sigma 8.


def descriptor(octave_img, blur_sigma, key_points):
    octave_img = cv.cvtColor(octave_img, cv.COLOR_BGR2GRAY)
    temp_scale = cv.copyMakeBorder(octave_img, 9, 8, 9, 8, cv.BORDER_REFLECT, None, 1)

    temp_m = np.zeros((octave_img.shape[0] + 15, octave_img.shape[1] + 15))
    temp_theta = np.zeros((octave_img.shape[0] + 15, octave_img.shape[1] + 15))
    for i in range(1, octave_img.shape[0] + 16):
        for j in range(1, octave_img.shape[1] + 16):
            temp_m[i - 1, j - 1] = ((temp_scale[i + 1, j] - temp_scale[i - 1, j]) ** 2 + (temp_scale[i, j + 1] - temp_scale[i, j - 1]) ** 2) ** 0.5
            t = np.degrees(math.atan2((temp_scale[i, j + 1] - temp_scale[i, j - 1]),
                            (temp_scale[i + 1, j] - temp_scale[i - 1, j])))
            if t < 0:
                t = t + 360
            temp_theta[i - 1, j - 1] = t

    #print('TEMP SCALE', temp_theta)
    descriptors = {}
    for loc in key_points:
        temp_bins = []
        gaussian_ker = gaussianKernel(16, 8)
        x = loc[0]
        y = loc[1]
        mat_m = get_submat(temp_m, (x - 8, y - 8), 16)
        mat_t = get_submat(temp_theta, (x - 8, y - 8), 16)
        for start_x in [0, 4, 8, 12]:
            for start_y in [0, 4, 8, 12]:
                sub_theta = get_submat(mat_m, (start_x, start_y), 4)
                sub_m = get_submat(mat_t, (start_x, start_y), 4)
                g_ker = get_submat(gaussian_ker, (start_x, start_y), 4)
                temp = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}

                for i in range(4):
                    for j in range(4):
                        if sub_theta[i, j] == 360:
                            temp[7] += sub_m[i, j] * g_ker[i, j]
                            continue
                        temp[sub_theta[i, j] // 45] += sub_m[i, j] * g_ker[i, j]
                for i in range(8):
                    temp_bins.append(temp[i])
        temp_bins = np.asarray(temp_bins)
        temp_bins = temp_bins/np.linalg.norm(temp_bins)
        descriptors[loc] = temp_bins
    return descriptors


#########################################################

#This part uses the function defined above on a given image
# It reads two images and makes their scale spaces. 4 octaves are constructed with each octave
# containing five images. The DoG scale space images are then calculated. The keypoints are then calculated



"""X = cv.imread('IMG0_001.jpg')
Y = cv.imread('IMG1_002.jpg')
print(X.shape)
print(Y.shape)
Y = cv.cvtColor(Y, cv.COLOR_BGR2GRAY)
X = cv.cvtColor(X, cv.COLOR_BGR2GRAY)
X = cv.resize(X, None, fx = 0.5, fy = 0.5, interpolation = cv.INTER_CUBIC)
#Y = cv.resize(Y, None, fx = 0.5, fy = 0.5, interpolation = cv.INTER_CUBIC)

X = cv.resize(X, (2 * X.shape[0], 2 * X.shape[1]))
Y = cv.resize(Y, (2 * Y.shape[0], 2 * Y.shape[1]))

a = np.sqrt(2)
sigma = 1/np.sqrt(2)
print('A ', a)
Xoctave1 = octave_gen(X, a, sigma, one_sigmas)
print('A ', a)
fig=plt.figure()
for i in range(len(Xoctave1)):
    fig.add_subplot(1, 5, i+1)
    plt.imshow(Xoctave1[i], cmap='gray')
print(X.shape)
plt.show()

for i in range(len(Xoctave1)):
    print('Octave 1')
    plt.imshow(Xoctave1[i], cmap = 'gray')
    plt.show()

Yoctave1 = octave_gen(Y, a, sigma, one_sigmas)
fig=plt.figure()
for i in range(len(Yoctave1)):
    fig.add_subplot(1, 5, i+1)
    plt.imshow(Yoctave1[i], cmap='gray')

plt.show()


temp_sigma = a**2 * sigma
Xx_size = int(X.shape[0]/2)
Xy_size = int(X.shape[1]/2)
X = cv.resize(X, None, fx = 0.5, fy = 0.5, interpolation = cv.INTER_CUBIC)
Xoctave2 = octave_gen(X, a, temp_sigma, two_sigmas)

fig=plt.figure()
for i in range(len(Xoctave2)):
    fig.add_subplot(1, 5, i+1)
    plt.imshow(Xoctave2[i], cmap='gray')
print(X.shape)
plt.show()
for i in range(len(Xoctave2)):
    print('Octave 2')
    plt.imshow(Xoctave2[i], cmap = 'gray')
    plt.show()

Yx_size = int(Y.shape[0]/2)
Yy_size = int(Y.shape[1]/2)
Y = cv.resize(Y, None, fx = 0.5, fy = 0.5, interpolation = cv.INTER_CUBIC)
#Yoctave2 = octave_gen(Y, a, temp_sigma, two_sigmas)

temp_sigma = a**4 * sigma
Xx_size = int(X.shape[0]/2)
Xy_size = int(X.shape[1]/2)
X = cv.resize(X, None, fx = 0.5, fy = 0.5, interpolation = cv.INTER_CUBIC)
Xoctave3 = octave_gen(X, a, temp_sigma, three_sigmas)

fig=plt.figure()
for i in range(len(Xoctave3)):
    fig.add_subplot(1, 5, i+1)
    plt.imshow(Xoctave3[i], cmap='gray')
print(X.shape)
plt.show()

for i in range(len(Xoctave3)):
    print('Octave 3')
    plt.imshow(Xoctave3[i], cmap = 'gray')
    plt.show()

Yx_size = int(Y.shape[0]/2)
Yy_size = int(Y.shape[1]/2)
Y = cv.resize(Y, None, fx = 0.5, fy = 0.5, interpolation = cv.INTER_CUBIC)
#Yoctave3 = octave_gen(Y, a, temp_sigma, three_sigmas)

temp_sigma = a**6 * sigma
Xx_size = int(X.shape[0]/2)
Xy_size = int(X.shape[1]/2)
X = cv.resize(X, None, fx = 0.5, fy = 0.5, interpolation = cv.INTER_CUBIC)
Xoctave4 = octave_gen(X, a, temp_sigma, four_sigmas)

fig=plt.figure()
for i in range(len(Xoctave4)):
    fig.add_subplot(1, 5, i+1)
    plt.imshow(Xoctave4[i], cmap='gray')
print(X.shape)
plt.show()
for i in range(len(Xoctave4)):
    print('Octave 4')
    plt.imshow(Xoctave4[i], cmap = 'gray')
    plt.show()

Yx_size = int(Y.shape[0]/2)
Yy_size = int(Y.shape[1]/2)
Y = cv.resize(Y, None, fx = 0.5, fy = 0.5, interpolation = cv.INTER_CUBIC)
#Yoctave4 = octave_gen(Y, a, temp_sigma, four_sigmas)
"""
Xoctave1 = []
Xoctave2 = []
Xoctave3 = []
Xoctave4 = []
Xoctave1.append(cv.imread('Figure_11.png'))
Xoctave1.append(cv.imread('Figure_12.png'))
Xoctave1.append(cv.imread('Figure_13.png'))
Xoctave1.append(cv.imread('Figure_14.png'))
Xoctave1.append(cv.imread('Figure_5.png'))

Xoctave2.append(cv.imread('Figure_21.png'))
Xoctave2.append(cv.imread('Figure_22.png'))
Xoctave2.append(cv.imread('Figure_23.png'))
Xoctave2.append(cv.imread('Figure_24.png'))
Xoctave2.append(cv.imread('Figure_25.png'))

Xoctave3.append(cv.imread('Figure_31.png'))
Xoctave3.append(cv.imread('Figure_32.png'))
Xoctave3.append(cv.imread('Figure_33.png'))
Xoctave3.append(cv.imread('34.png'))
Xoctave3.append(cv.imread('Figure_35.png'))

Xoctave4.append(cv.imread('Figure_41.png'))
Xoctave4.append(cv.imread('Figure_42.png'))
Xoctave4.append(cv.imread('Figu4_3.png'))
Xoctave4.append(cv.imread('Figure_44.png'))
Xoctave4.append(cv.imread('Figure_45.png'))

Xoctave1_dog = dog_gen(Xoctave1)

"""for i in range(len(Xoctave1_dog)):
    print('Octave 1')
    plt.imshow(Xoctave1_dog[i], cmap = 'gray')
    plt.show()"""

Xoctave2_dog = dog_gen(Xoctave2)

"""for i in range(len(Xoctave1_dog)):
    print('Octave 2')
    plt.imshow(Xoctave2_dog[i], cmap = 'gray')
    plt.show()
"""
Xoctave3_dog = dog_gen(Xoctave3)

"""for i in range(len(Xoctave1_dog)):
    print('Octave 3')
    plt.imshow(Xoctave3_dog[i], cmap = 'gray')
    plt.show()"""

Xoctave4_dog = dog_gen(Xoctave4)

"""for i in range(len(Xoctave1_dog)):
    print('Octave 4')
    plt.imshow(Xoctave4_dog[i], cmap = 'gray')
    plt.show()"""


Xoctave1_key_dog2 = key_point(Xoctave1_dog[1], Xoctave1_dog[2], Xoctave1_dog[0])
Xoctave1_key_dog3 = key_point(Xoctave1_dog[2], Xoctave1_dog[3], Xoctave1_dog[1])

Xoctave2_key_dog2 = key_point(Xoctave2_dog[1], Xoctave2_dog[2], Xoctave2_dog[0])
Xoctave2_key_dog3 = key_point(Xoctave2_dog[2], Xoctave2_dog[3], Xoctave2_dog[1])

Xoctave3_key_dog2 = key_point(Xoctave3_dog[1], Xoctave3_dog[2], Xoctave3_dog[0])
Xoctave3_key_dog3 = key_point(Xoctave3_dog[2], Xoctave3_dog[3], Xoctave3_dog[1])

Xoctave4_key_dog2 = key_point(Xoctave4_dog[1], Xoctave4_dog[2], Xoctave4_dog[0])
Xoctave4_key_dog3 = key_point(Xoctave4_dog[2], Xoctave4_dog[3], Xoctave4_dog[1])


Yoctave1 = []
Yoctave2 = []
Yoctave3 = []
Yoctave4 = []
Yoctave1.append(cv.imread('Figure_11.png'))
Yoctave1.append(cv.imread('Figure_12.png'))
Yoctave1.append(cv.imread('Figure_13.png'))
Yoctave1.append(cv.imread('Figure_14.png'))
Yoctave1.append(cv.imread('Figure_5.png'))

Yoctave2.append(cv.imread('Figure_21.png'))
Yoctave2.append(cv.imread('Figure_22.png'))
Yoctave2.append(cv.imread('Figure_23.png'))
Yoctave2.append(cv.imread('Figure_24.png'))
Yoctave2.append(cv.imread('Figure_25.png'))

Yoctave3.append(cv.imread('Figure_31.png'))
Yoctave3.append(cv.imread('Figure_32.png'))
Yoctave3.append(cv.imread('Figure_33.png'))
Yoctave3.append(cv.imread('34.png'))
Yoctave3.append(cv.imread('Figure_35.png'))

Yoctave4.append(cv.imread('Figure_41.png'))
Yoctave4.append(cv.imread('Figure_42.png'))
Yoctave4.append(cv.imread('Figu4_3.png'))
Yoctave4.append(cv.imread('Figure_44.png'))
Yoctave4.append(cv.imread('Figure_45.png'))

Yoctave1_dog = dog_gen(Yoctave1)

Yoctave2_dog = dog_gen(Yoctave2)

Yoctave3_dog = dog_gen(Yoctave3)

Yoctave4_dog = dog_gen(Yoctave4)


Yoctave1_key_dog2 = key_point(Yoctave1_dog[1], Yoctave1_dog[2], Yoctave1_dog[0])
Yoctave1_key_dog3 = key_point(Yoctave1_dog[2], Yoctave1_dog[3], Yoctave1_dog[1])

Yoctave2_key_dog2 = key_point(Yoctave2_dog[1], Yoctave2_dog[2], Yoctave2_dog[0])
Yoctave2_key_dog3 = key_point(Yoctave2_dog[2], Yoctave2_dog[3], Yoctave2_dog[1])

Yoctave3_key_dog2 = key_point(Yoctave3_dog[1], Yoctave3_dog[2], Yoctave3_dog[0])
Yoctave3_key_dog3 = key_point(Yoctave3_dog[2], Yoctave3_dog[3], Yoctave3_dog[1])

Yoctave4_key_dog2 = key_point(Yoctave4_dog[1], Yoctave4_dog[2], Yoctave4_dog[0])
Yoctave4_key_dog3 = key_point(Yoctave4_dog[2], Yoctave4_dog[3], Yoctave4_dog[1])




Xfirst_1 = descriptor(Xoctave1[1], np.sqrt(1/2) * np.sqrt(2), Xoctave1_key_dog2)
Xfirst_3 = descriptor(Xoctave1[3], np.sqrt(1/2) * np.sqrt(2)**3, Xoctave1_key_dog3)

Xsecond_1 = descriptor(Xoctave2[1], np.sqrt(1/2) * np.sqrt(2)**3, Xoctave2_key_dog2)
Xsecond_3 = descriptor(Xoctave2[3], np.sqrt(1/2) * np.sqrt(2)**5, Xoctave2_key_dog3)

Xthird_1 = descriptor(Xoctave3[1], np.sqrt(1/2) * np.sqrt(2)** 5, Xoctave3_key_dog2)
Xthird_3 = descriptor(Xoctave3[3], np.sqrt(1/2) * np.sqrt(2)**7, Xoctave3_key_dog3)

Xfourth_1 = descriptor(Xoctave4[1], np.sqrt(1/2) * np.sqrt(2)**7, Xoctave4_key_dog2)
Xfourth_3 = descriptor(Xoctave4[3], np.sqrt(1/2) * np.sqrt(2)**9, Xoctave4_key_dog3)




Yfirst_1 = descriptor(Yoctave1[1], one_sigmas[1], Yoctave1_key_dog2)
Yfirst_3 = descriptor(Yoctave1[3], one_sigmas[3], Yoctave1_key_dog3)

Ysecond_1 = descriptor(Yoctave2[1], two_sigmas[1], Yoctave2_key_dog2)
Ysecond_3 = descriptor(Yoctave2[3], two_sigmas[3], Yoctave2_key_dog3)

Ythird_1 = descriptor(Yoctave3[1], three_sigmas[1], Yoctave3_key_dog2)
Ythird_3 = descriptor(Yoctave3[3], three_sigmas[3], Yoctave3_key_dog3)

Yfourth_1 = descriptor(Yoctave4[1], four_sigmas[1], Yoctave4_key_dog2)
Yfourth_3 = descriptor(Yoctave4[3], four_sigmas[3], Yoctave4_key_dog3)


for posX in Xfirst_1.keys():
    if posX[0] > Xoctave1[1].shape[0]/4 and posX[0] < Xoctave1[1].shape[0] * 3/4 and posX[1] > Xoctave1[1].shape[1]/4 and posX[1] < Xoctave1[1].shape[1] * 3/4:
        desX = np.asarray(Xfirst_1[posX])
        temp = (0, 0)
        val = 10000000000
        for posY in Yfirst_1.keys():
            desY = np.asarray(Yfirst_1[posY])
            t = np.linalg.norm(desX - desY)
            if t < val:
                val = t
                temp = posY

        tempXimg = Xoctave1[1]
        tempYimg = Yoctave1[1]
        print('posX ', posX)
        print('posY ', temp, 'des ', Yfirst_1[temp])
        cv.circle(tempXimg, posX, 2, (0, 255, 0), -1)
        cv.circle(tempYimg, temp, 2, (0, 255, 0), -1)
        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        plt.imshow(tempXimg, cmap='gray')
        fig.add_subplot(1, 2, 2)
        plt.imshow(tempYimg, cmap='gray')
        plt.show()


"""for i in range(20):
    posX, desX = random.choice(list(Xfirst_1.items()))
    desX = np.asarray(desX)
    temp = (0, 0)
    val = 10000000000
    c
    cv.circle(Xoctave1[1], posX, 2, (0, 255, 0), -1)
    cv.circle(Yoctave1[1], temp, 2, (0, 255, 0), -1)



fig=plt.figure()
fig.add_subplot(1, 2, 1)
plt.imshow(Xoctave1[1], cmap='gray')
fig.add_subplot(1, 2, 2)
plt.imshow(Yoctave1[1], cmap='gray')

plt.show()
"""


"""cv2.circle(img,(row, col), 5, (0,255,0), -1)"""







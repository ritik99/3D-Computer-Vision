import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math
import skimage.feature

#This method finds the SIFT features
def detectKeyPoints(img1):
    sift = cv.xfeatures2d.SIFT_create()
    keyPoints, descriptors = sift.detectAndCompute(img1, None)

    return (keyPoints, descriptors)


'''This method is used for keypoint matching. It uses the detectKeyPoints function. It returns the fundamental matrix'''

def keyPointMatching(img1, img2):


    keyPoints1, descriptors1 = detectKeyPoints(img1)
    keyPoints2, descriptors2 = detectKeyPoints(img2)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    pts1 = []
    pts2 = []
    good = []
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            good.append(m)
            pts2.append(keyPoints2[m.trainIdx].pt)
            pts1.append(keyPoints1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv.findFundamentalMat(pts1,pts2)
    #F = F.T
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    return (F, mask, pts1, pts2)


#This method can be used to draw circles around the daisy descriptors found.
def compute_and_drawlines(img1, img2, pts1, pts2, F):
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    print(pts1.shape)
    lines1 = lines1.reshape(-1,3)
    print('Shape of lines: ', lines1.shape)
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 2,F)
    lines2 = lines2.reshape(-1,3)
    ''' img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    '''for r,pt1,pt2 in zip(lines1,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)

    imgplot = plt.imshow(img1)
    #plt.show()
    imgplot = plt.imshow(img2)
    #plt.show()'''
    return (lines1, lines2)

#Please change the below lines to try on different image
img1 = cv.imread('im3L.bmp',0)  #queryimage # left image
img2 = cv.imread('im3R.bmp',0) #trainimage # right image
img1_color = cv.imread('im3L.bmp')  #queryimage # left image
img2_color = cv.imread('im3R.bmp') #trainimage # right image
curr = 'im3L.bmp'
img1 = cv.resize(img1, None, fx = 0.5, fy = 0.5, interpolation = cv.INTER_CUBIC)
img2 = cv.resize(img2, None, fx = 0.5, fy = 0.5, interpolation = cv.INTER_CUBIC)
img1_color = cv.resize(img1_color, None, fx = 0.5, fy = 0.5, interpolation = cv.INTER_CUBIC)
img2_color = cv.resize(img2_color, None, fx = 0.5, fy = 0.5, interpolation = cv.INTER_CUBIC)
offset = {'im3L.bmp': (240, 119), 'im1L.bmp': (0, 0)}

#kpts1, des1 = detectKeyPoints(img1)
#kpts2, des2 = detectKeyPoints(img2)
F, mask, pts1, pts2 = keyPointMatching(img1, img2)
#lines1, lines2 = compute_and_drawlines(img1, img2, pts1, pts2, F)

padded_img1 = cv.copyMakeBorder(img1,50,50,50,50,cv.BORDER_REPLICATE)
padded_img2 = cv.copyMakeBorder(img2,50,50,50,50,cv.BORDER_REPLICATE)


#The daisy descriptors are calculated for each point here.

daisy_descs1 = skimage.feature.daisy(visualize=False, image=padded_img1, step=1, radius=50, rings=1, histograms=16, orientations=8)
daisy_descs2 = skimage.feature.daisy(visualize=False, image=padded_img2, step=1, radius=50, rings=1, histograms=16, orientations=8)

print('Shape of image ', img1.shape)
print('Shape of daisy : ', daisy_descs1.shape)
print('Shape of daisy : ', daisy_descs2.shape)


#This is the main function where the epilines are calculated and then the mapping for each point is found
def find_mapping(img1, img2, daisy_descs1, daisy_descs2, F):
    mapping = {}
    for y in range(img2.shape[0]):
        for x in range(img2.shape[1]):
            pts = np.array((y, x))
            lines1 = cv.computeCorrespondEpilines(pts.reshape(-1,1,2), 2,F)
            lines1 = lines1.reshape(-1,3)
            possible = lie_on_ine(lines1, img2)
            min_dist = 1000000000
            desc_length = daisy_descs1.shape[2]
            possible_descs = np.zeros((len(possible), desc_length))
            #print('Length of possible ', len(possible))
            if len(possible) == 0:
                if curr in offset.keys():
                    if x >= offset[curr][0]:
                        x_val = mapping[y, offset[curr][0]][0] - offset[curr][1]
                        #print('No options here')
                        if x != 0:
                            mapping[(y, x)] = (x_val, mapping[(y, x - 1)][1] + 1)
                        else:
                            mapping[(y, x)] = (x_val, mapping[(y - 1, x)][1])
                        #print('mapping ', (y, x), ' ', mapping[(y, x)])
                        continue
                else:
                    if x != 0:
                        mapping[(y, x)] = (mapping[(y, x - 1)][0], mapping[(y, x - 1)][1] + 1) 
                    elif x == 0 and y == 0:
                        mapping[(y, x)] = (0, 0)
                    else:
                        mapping[(y, x)] = (mapping[(y-1, x)][0] + 1, mapping[(y - 1, x)][1])

            count = 0
            for pt in possible:
                possible_descs[0] = daisy_descs2[pt[0], pt[1]]
                count += 1

            temp = possible_descs - daisy_descs1[y, x]
            temp_norm = np.linalg.norm(temp, axis=1)
            #print('Length of possible ',len(possible))
            #print(temp_norm)
            pt = possible[np.argmin(temp_norm)]
            mapping[(y, x)] = (pt[1] + y, pt[0])
            #mapping[(i, j)] = possible[0]
            #print('mapping ', (y, x), ' ', mapping[(y, x)])
            #print((i, j), ' ', mapping[(i,j)], ' ', np.argmin(temp_norm))
            '''for pt in possible:
                if np.linalg.norm(daisy_descs1[i, j] - daisy_descs2[pt[0], pt[1]]) < min_dist:
                    min_dist = np.linalg.norm(daisy_descs1[i, j] - daisy_descs2[pt[0], pt[1]])
                    mapping[(i, j)] = (pt[0], pt[1])'''
        print('Done with ', (y, x))
    return mapping

#This is used for plotting the images in new co-ordinates

def plot(img1, img2, mapping):
    canvas = np.zeros((600, 600, 3), dtype = np.uint8)
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            print('imag val ', img1_color[i, j])
            canvas[mapping[(i, j)]] = img1_color[i, j]
    canvas_uint = canvas.astype(np.uint8)
    cv.imwrite('output.jpg', canvas_uint)
    cv.imshow('fig', canvas)
    cv.waitKey(0)


#This function finds the points that lie on the epiline and by solving for y by putting every value x
def lie_on_ine(coeff, img2):
    a, b, c = coeff[0]
    possible = []
    for i in range(img2.shape[1]):
        if int((-c - a * i)/b) < img2.shape[0] and int((-c - a * i)/b) >= 0:
            possible.append((int((-c - a * i)/b), i))
    #print('Length of possible ', len(possible))
    return possible
            

mapping = find_mapping(img1, img2, daisy_descs1, daisy_descs2, F)
plot(img1, img2, mapping)






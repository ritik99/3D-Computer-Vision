"""
Author: Ritik Dutta
Roll Number: 16110053

This program has the following methods:
1. gau(x, y, sigma) - Returns the output of applying the Gaussian function for a particular co-ordinate.
Its inputs are the (x,y) position in the matrix, and the sigma of the Gaussian function

2. gaussian_kernel(size, sigma) - Returns the normalised Gaussian filter of specified size. Its inputs are the size
of the filter and sigma of the Gaussian function

3. diff_of_gaussian(size, sigma1, sigma2) - Returns the Difference of Gaussian filter. Its inputs are:
a. size: Size of the kernel
b. sigma1: Sigma for one of the Gaussians
c. sigma2: Sigma for the other Gaussian

4. convolve(X, M) - This function performs the convolution operation. Its inputs are the image (in the form of
matrix) on which convolution has to be performed, and the filter which is used to perform convolution.

5. detectEdge(X, threshold) - This function detects the zero crossings in the convolved image. Its input is
the input image.

"""



import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
np.set_printoptions(threshold=np.inf, precision = 3)
from mpl_toolkits import mplot3d


def gau(x, y, sigma): 
    return 1/(2*np.pi*(sigma)**2) * np.exp(-(x**2 + y**2)/(2*(sigma)**2))  #Implements the Gaussian function

def gaussian_kernel(size, sigma):   #This function outputs the Gaussian Kernel
    M = np.zeros((size,size), dtype = np.float32) 
    s = 0
    for i in range(-(size//2), size//2 + 1):  #Calculates the value of each element of the kernel by calling gau()
        for j in range(-(size//2), size//2 + 1):
            M[i+size//2, j + size//2] = gau(i, j, sigma)
            s = s + M[i+size//2, j + size//2]

    for i in range(M.shape[0]):     #This part normalises the kernel
        for j in range(M.shape[1]):
            M[i, j] /= s
    M = M.astype(np.float32)
    print(M)    #prints the kernel. Comment the line if not needed
    return M

def diff_of_gaussian(size, sigma1, sigma2):    #This function outputs the difference of Gaussian kernel
    X = np.zeros((size,size))#, dtype = np.float32)
    X = gaussian_kernel(11, 1.1)
    
    Y = np.zeros((size,size))#, dtype = np.float32)
    Y = gaussian_kernel(11, 2.5)

    M = Y - X
    M = M.astype(np.float32)
    print(M)    #prints the kernel. Comment the line if not needed
    return M
    

#This function performs the convolution. It performs the padding operation as well.

def convolve(X, M): 
    padding_v = M.shape[0]//2
    tempX = X.shape[0]
    tempY = X.shape[1]
    X_copy = np.zeros((padding_v + tempX + padding_v, padding_v + tempY + padding_v))
    for i in range(padding_v, padding_v + tempX):       #This part performs the padding operation
        for j in range(padding_v, padding_v + tempY):
            X_copy.itemset((i, j), X[i - padding_v, j - padding_v])
    #print(X_copy)

    Y = np.zeros((tempX, tempY), dtype = np.float32)
    posX = 0
    posY = 0
    for i in range(0, X_copy.shape[0] - M.shape[0]):        #This part performs the actual convolution
        for j in range(0, X_copy.shape[1] - M.shape[1]):
            t = X_copy[np.ix_([k for k in range(i, M.shape[0] + i)], [l for l in range(j, M.shape[1] + j)])]
            tt = 0
            for k in range(M.shape[0]):
                for l in range(M.shape[1]):
                    tt = tt + M[k, l]*t[k, l]
            Y[posX, posY] = tt
            posY += 1
        posY = 0
        posX += 1
    return Y

#Performs the edge detection operation
def detectEdge(X):
    M = diff_of_gaussian(11, 1.1, 2.5)
    Z = convolve(X, M)

    Z = Z.astype(np.float32)
    Q = np.zeros((Z.shape[0], Z.shape[1]), dtype = np.uint8)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            if Z[i, j] < 0:     #This part detects zero crossing using 4-neighbours of a pixel. 
                if Z[i-1, j] > 0 or Z[i+1, j] > 0 or Z[i, j-1] > 0 or Z[i, j+1] > 0:    #If a pixel value is negative and one of its neighbours is positive
                    Q[i,j] = 1              #The value of the pixel is set to 1.
    
    return Q
                

X = cv.imread('butterfly.jpg')
X = cv.cvtColor(X, cv.COLOR_BGR2GRAY)

#M = gaussian_kernel(9, 1)   #Uncomment to print Gaussian Kernel with Std. Deviation 1. Uncomment the M = convolve(X, M) as well to output the convolved image

#M = gaussian_kernel(9, 3)  #Uncomment to print Gaussian Kernel with Std. Deviation 3. Uncomment the M = convolve(X, M) as well to output the convolved image

#M = gaussian_kernel(9, 20)  #Uncomment to print Gaussian Kernel with Std. Deviation 20. Uncomment the M = convolve(X, M) as well to output the convolved image

#M = convolve(X, M)

M = detectEdge(X)
 
#M = diff_of_gaussian(11, 1.1, 2.5)
#M = convolve(X, M)


plt.imshow(M, cmap = 'gray')
plt.show()




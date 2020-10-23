import cv2 as cv
import numpy as np
import os
from math import sqrt

class Solution():
    """
    The additional parameter k is added into a Gaussian filter
    In order to achieve a high-boosting result
    original transform matrix: d
    proposed transform matrix: (1-d)*k+1
    """
    def __init__(self):
        super(Solution).__init__()
    
    def imread(self, path):
        img = cv.imread(path)
        img_grayscaled = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return img, img_grayscaled
    
    def imsave(self, path, img):
        cv.imwrite(path, img)

    def make_transform_matrix(self, img, s1, d):
        transfor_matrix = np.zeros(img.shape)
        center_point = tuple(map(lambda x:(x-1)/2,s1.shape))
        for i in range(transfor_matrix.shape[0]):
            for j in range(transfor_matrix.shape[1]):
                def cal_distance(pa, pb):
                    dis = sqrt((pa[0]-pb[0])**2+(pa[1]-pb[1])**2)
                    return dis
                dis = cal_distance(center_point,(i,j))
                transfor_matrix[i,j] = np.exp(-(dis**2)/(2*(d**2)))
        return transfor_matrix

    def GaussianLowFilter_withHighBoost(self, img, d, k):
        # GLPF, parameters: img array and D, k is the boosting parameter, default 1
        h, w = img.shape[:2]
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        s1 = np.log(np.abs(fshift))
        d_matrix = self.make_transform_matrix(img, s1, d)
        # here I mimic the laplacian's way to perform the mask
        new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift + fshift*k*(1-d_matrix))))
        return new_img

def main():
    path1 = './blurry-moon.tif'
    s = Solution()
    im1, im1_grayscale = s.imread(path1)
    # question a
    out1 = s.GaussianLowFilter_withHighBoost(im1_grayscale, d = 10, k = 1)
    s.imsave('./result_a.jpg', out1)
    # question b
    out2 = s.GaussianLowFilter_withHighBoost(im1_grayscale, d = 10, k = 3)
    s.imsave('./result_b.jpg', out1)

if __name__ == "__main__":
    main()
import cv2 as cv
import numpy as np
import os
from math import sqrt

class Solution():
    """
    LPF based on fourier transformation (Frequency-domaint filters)
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

    def GaussianLowFilter(self, img, d):
        # GLPF, parameters: img array and D
        h, w = img.shape[:2]
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        s1 = np.log(np.abs(fshift))
        d_matrix = self.make_transform_matrix(img, s1, d)
        new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*d_matrix)))
        return new_img

    def make_transform_matrix_butterworth(self, img, s1, d, n):
        transfor_matrix = np.zeros(img.shape)
        center_point = tuple(map(lambda x:(x-1)/2,s1.shape))
        for i in range(transfor_matrix.shape[0]):
            for j in range(transfor_matrix.shape[1]):
                def cal_distance(pa,pb):
                    dis = sqrt((pa[0]-pb[0])**2+(pa[1]-pb[1])**2)
                    return dis
                dis = cal_distance(center_point,(i,j))
                transfor_matrix[i,j] = 1/((1+(dis/d))**n)
        return transfor_matrix

    def ButterworthPassFilter(self, img, d, n):
        # Butterworth LPF
        h, w = img.shape[:2]
        mask = 255 * np.ones((h, w))
        img = mask - img
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        s1 = np.log(np.abs(fshift))
        d_matrix = self.make_transform_matrix_butterworth(img, s1, d, n)
        new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*d_matrix)))
        ret, thresh1 = cv.threshold(new_img, 65, 255, cv.THRESH_BINARY)
        out = mask - thresh1
        return out

def main():
    s = Solution()
    path1 = './testpattern1024.tif'
    path2 = './checkerboard1024-shaded.tif'
    im1, im1_gray = s.imread(path1)
    # question a
    out_1 = s.GaussianLowFilter(im1_gray, d = 5)
    s.imsave('./result_a.jpg', out_1)

    # question b
    out_2 = s.ButterworthPassFilter(im1_gray, d = 1, n = 1)
    s.imsave('./result_b.jpg', out_2)

    # question c
    im2, im2_gray = s.imread(path2)
    background_im2 = s.GaussianLowFilter(im2_gray, d = 3)
    s.imsave('./result_c_background.jpg', background_im2)
    s.imsave('./result_c.jpg', 255*(im2_gray / background_im2))

if __name__ == "__main__":
    main()
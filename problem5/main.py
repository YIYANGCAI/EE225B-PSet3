import cv2 as cv
import numpy as np
import os
from math import sqrt

class Solution():
    """
    HPF in frequency domain
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
                transfor_matrix[i,j] = np.exp(-(2*(d**2)/(dis**2)))
        return transfor_matrix

    def GaussianLowFilter(self, img, d):
        # GLPF, parameters: img array and D
        h, w = img.shape[:2]
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        s1 = np.log(np.abs(fshift))
        d_matrix = self.make_transform_matrix(img, s1, d)
        new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*d_matrix)))
        new_img = new_img.astype('uint8')
        ret1, th1 = cv.threshold(new_img, 0, 255, cv.THRESH_OTSU)
        return th1

    def make_transform_matrix_butterworth(self, img, s1, d, n):
        transfor_matrix = np.zeros(img.shape)
        center_point = tuple(map(lambda x:(x-1)/2,s1.shape))
        for i in range(transfor_matrix.shape[0]):
            for j in range(transfor_matrix.shape[1]):
                def cal_distance(pa,pb):
                    dis = sqrt((pa[0]-pb[0])**2+(pa[1]-pb[1])**2)
                    return dis
                dis = cal_distance(center_point,(i,j))
                transfor_matrix[i,j] = 1/((1+(d/dis))**n)
        return transfor_matrix

    def ButterworthPassFilter(self, img, d, n):
        # Butterworth
        h, w = img.shape[:2]
        mask = 255 * np.ones((h, w))
        img = mask - img
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        s1 = np.log(np.abs(fshift))
        d_matrix = self.make_transform_matrix_butterworth(img, s1, d, n)
        _new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*d_matrix)))
        new_img = np.fft.ifft2(np.fft.ifftshift(fshift*d_matrix))
        thresh1 = np.zeros((h,w))
        # do the proposed threshold operation
        for i in range(h):
            for j in range(w):
                if new_img[i,j] < 0: thresh1[i,j] = 255
                else: thresh1[i,j] = 0
        return _new_img, thresh1

    def make_transform_matrix_laplacian(self, img, s1):
        h, w = img.shape[:2]
        transfor_matrix = np.zeros(img.shape)
        center_point = tuple(map(lambda x:(x-1)/2,s1.shape))
        for i in range(transfor_matrix.shape[0]):
            for j in range(transfor_matrix.shape[1]):
                def cal_distance(pa,pb):
                    dis_s = (pa[0]-pb[0])**2+(pa[1]-pb[1])**2
                    dis = sqrt((pa[0]-pb[0])**2+(pa[1]-pb[1])**2)
                    return dis_s, dis
                dis_s, dis = cal_distance(center_point,(i,j))
                transfor_matrix[i,j] = (4 * 3.14 * 3.14 * dis_s) / (h*w)
        return transfor_matrix

    def LaplacianFilter(self, img):
        # Butterworth LPF
        h, w = img.shape[:2]
        mask = 255 * np.ones((h, w))
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        s1 = np.log(np.abs(fshift))
        d_matrix = self.make_transform_matrix_laplacian(img, s1)
        new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift + 4*fshift*d_matrix)))
        #new_img = new_img.astype('uint8')
        #ret1, th1 = cv.threshold(img.astype('uint8'), 0, 255, cv.THRESH_OTSU)
        #ret, thresh1 = cv.threshold(new_img, 11, 255, cv.THRESH_BINARY)
        #out = mask - thresh1
        return new_img
    
    def Butterworth_Chest(self, img, d, n):
        # Butterworth
        h, w = img.shape[:2]
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        s1 = np.log(np.abs(fshift))
        d_matrix = self.make_transform_matrix_butterworth(img, s1, d, n)
        new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(0.5+1.5*fshift*d_matrix)))
        new_img_1 = np.abs(np.fft.ifft2(np.fft.ifftshift(0.5+3*fshift*d_matrix)))
        return new_img, new_img_1

    def histEqual4e(self, img):
        init_hist = np.zeros(256)
        mapping = np.zeros(256)
        data = img.flatten()
        h, w = img.shape[:2]
        total = h * w
        new_img = np.zeros((h, w))
        for pixel in list(data):
            init_hist[pixel] = init_hist[pixel] + 1
        for rk in range(256):
            accumulate = 0
            for _index in range(rk):
                accumulate += init_hist[_index]
            sk = int(256*accumulate/total)
            mapping[rk] = sk
        for i in range(h):
            for j in range(w):
                prev_v = img[i,j]
                equa_v = mapping[prev_v]
                new_img[i,j] = equa_v
        return new_img

def main():
    s = Solution()
    path1 = './Fig0457(a)(thumb_print).tif'
    path2 = './blurry-moon.tif'
    path3 = './Fig0459(a)(orig_chest_xray).tif'
    im1, im1_gray = s.imread(path1)
    im2, im2_gray = s.imread(path2)
    im3, im3_gray = s.imread(path3)

    # question a
    filtered, out_1 = s.ButterworthPassFilter(im1_gray, d = 50, n = 1)
    s.imsave('./result_a_1.jpg', filtered)
    s.imsave('./result_a_2.jpg', out_1)

    # question b & c, proposed function of question b is self.LaplacianFilter()
    out_2 = s.LaplacianFilter(im2_gray)
    s.imsave('./result_bc.jpg', out_2)

    # question d
    out_3, out_4 = s.Butterworth_Chest(im3_gray, d = 40, n = 1)
    s.imsave('./result_d_1.jpg', out_3)
    s.imsave('./result_d_2.jpg', out_4)
    out_5 = s.histEqual4e(out_3.astype('uint8'))
    s.imsave('./result_d_3.jpg', out_5)

if __name__ == "__main__":
    main()
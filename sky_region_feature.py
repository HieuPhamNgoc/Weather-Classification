import cv2 as cv
import numpy as np
from scipy.ndimage import gaussian_filter

class sky_region:
    def __init__(self, img, regionX, regionY):
        self.img = img.astype('float')
        self.regionX = regionX
        self.regionY = regionY

    def sky_area(self, image, regionX, regionY):
        # Now sky area is 2d vector
        n_segments = len(regionX)
        sky_area = image[regionY[0], regionX[0]]
        for i in range(1, n_segments):
            sky_area = np.concatenate([sky_area, image[regionY[i], regionX[i]]])

        return sky_area

    def colorfulness(self, w1 = 1, w2 = 0.3):
        image_lab = cv.cvtColor(self.img.astype('uint8'), cv.COLOR_RGB2LAB)
        a = image_lab[:,:,1].astype('float')
        b = image_lab[:,:,2].astype('float')
        a = self.sky_area(a, self.regionX, self.regionY)
        b = self.sky_area(b, self.regionX, self.regionY)

        (aMean, aStd) = (np.mean(a), np.std(a))
        (bMean, bStd) = (np.mean(b), np.std(b))

        stdRoot = np.sqrt((aStd ** 2) + (bStd ** 2))
        meanRoot = np.sqrt((aMean ** 2) + (bMean ** 2))
        colorfulness =  w1 * stdRoot + (w2 * meanRoot)
        return colorfulness

    def intensity(self):
        sky_area = self.sky_area(self.img, self.regionX, self.regionY)
        R = sky_area[:, 0].astype('float')
        G = sky_area[:, 1].astype('float')
        B = sky_area[:, 2].astype('float')
        
        i = np.mean((R + G + B)/3)
        return i

    def avg_gradient(self):
        gray = cv.cvtColor(self.img.astype('uint8'), cv.COLOR_RGB2GRAY)
        sobelx = cv.Sobel(gray,cv.CV_64F,1,0)
        sobely = cv.Sobel(gray,cv.CV_64F,0,1)
        
        sqrt_gradient_sum = np.sqrt((sobelx ** 2) + (sobely ** 2))

        sky_region_gradient = self.sky_area(sqrt_gradient_sum, self.regionX, self.regionY)
        mean_lap = np.mean(sky_region_gradient)
        return mean_lap
        
    def contrast_energy(self, k = 0.1, tau_L = 0.2353, tau_a = 0.2287, tau_b = 0.0528):
        image_lab = cv.cvtColor(self.img.astype('uint8'), cv.COLOR_RGB2LAB)
        L = image_lab[:,:,0].astype('float')
        a = image_lab[:,:,1].astype('float')
        b = image_lab[:,:,2].astype('float')

        Lhh = gaussian_filter(L, sigma=4, order=[2, 0], mode='reflect')
        Lhv = gaussian_filter(L, sigma=4, order=[0, 2], mode='reflect')

        ahh = gaussian_filter(a, sigma=4, order=[2, 0], mode='reflect')
        ahv = gaussian_filter(a, sigma=4, order=[0, 2], mode='reflect')

        bhh = gaussian_filter(b, sigma=4, order=[2, 0], mode='reflect')
        bhv = gaussian_filter(b, sigma=4, order=[0, 2], mode='reflect')

        ZL = np.sqrt(Lhh ** 2 + Lhv ** 2)
        Za = np.sqrt(ahh ** 2 + ahv ** 2)
        Zb = np.sqrt(bhh ** 2 + bhv ** 2)

        ZL_sky = self.sky_area(ZL, self.regionX, self.regionY)
        Za_sky = self.sky_area(Za, self.regionX, self.regionY)
        Zb_sky = self.sky_area(Zb, self.regionX, self.regionY)

        alpha = np.max([np.max(ZL_sky), np.max(Za_sky), np.max(Zb_sky)])

        CEL = np.mean(alpha * ZL_sky / (ZL_sky + alpha * k) - tau_L)
        CEa = np.mean(alpha * Za_sky / (Za_sky + alpha * k) - tau_a)
        CEb = np.mean(alpha * Zb_sky / (Zb_sky + alpha * k) - tau_b)

        return CEL, CEa, CEb
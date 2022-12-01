import cv2 as cv
import numpy as np

class ROI_feature:
    '''
    Reference: https://www.mrt.kit.edu/z/publ/download/Roser_al2008iv.pdf
    '''

    def __init__(self):
        pass

    def image_rgb(self,img_path):
        '''
        Reads and returns RGB image from the path
        '''
        img = cv.imread(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        return img

    def image_resize(self,img, scale = 1, p = 0.5, height = 576, width=720):
        '''
        Resize input image, by scaling original image or by specific width and height
        '''
        if scale == 1:
            w = int(img.shape[1] * p)
            h = int(img.shape[0] * p)
            resized = cv.resize(img, (w, h), interpolation=cv.INTER_AREA)
            return resized
        else:
            resized = cv.resize(img, (width, height), interpolation = cv.INTER_AREA)
            return resized

    def image_brightness(self, img):
        r = img[:,:,0]
        g = img[:,:,1]
        b = img[:,:,2]
        return np.mean(r), np.mean(g), np.mean(b),

    def image_contrast(self, img):
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        max = np.max(gray)
        min = np.min(gray)
        if (max + min) == 0:
            return 0
        else:
            contrast = np.round((max - min)/(max+min), 5)
            return contrast

    def image_sharpness(self, img):
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        gX = cv.Sobel(gray, cv.CV_64F, 1, 0)
        gY = cv.Sobel(gray, cv.CV_64F, 0, 1)
        h, w = gX.shape
        sharpness = np.sum(np.sqrt(gX ** 2 + gY ** 2))/(h*w)

        return sharpness

    def hs(self,img):
        hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
        h = hsv[:,:,0]
        s = hsv[:,:,1]
        mean_h = np.mean(h)
        mean_s = np.mean(s)
        return mean_h, mean_s

    def combine_feature_one_roi(self,img):
        r, g, b = self.image_brightness(img)
        r = r.ravel()
        g = g.ravel()
        b = b.ravel()
        contrast = np.array(self.image_contrast(img))
        sharpness = np.array(self.image_sharpness(img))
        hue, sat = self.hs(img)
        hue = np.array(hue)
        sat = np.array(sat)
        v = np.concatenate(r, g, b, contrast, sharpness, hue, sat)
        return v

    def combine_feature_image(self, img, v):
        h, w, _ = img.shape
        w_range = range(w)
        h_range = range(h)

        w_chunks = np.array_split(w_range, 4)
        h_chunks = np.array_split(h_range, 3)
        for hc in h_chunks:
            for wc in w_chunks:
                crop = img[hc][:, wc]
                v = np.concatenate(v, self.combine_feature_one_roi(crop))

        return v

    
    
        





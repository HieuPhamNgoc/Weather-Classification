import cv2 as cv
import numpy as np

class ensemble_method_feature:
    '''
    Reference: https://ieeexplore-ieee-org.libproxy.aalto.fi/document/8704783
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
        

    def hsv(self, img):
        '''
        Takes an RGB image and return hue, saturation, value (HSV)
        '''
        hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)

        h = hsv[:,:,0]
        s = hsv[:,:,1]
        v = hsv[:,:,2]

        return h,s,v
    
    def gradient_magnitude(self, img):
        '''
        Turns RGB image to grayscale image and returns the gradient magnitude in both x and y direction
        '''
        # compute gradients along the x and y axis, respectively
        lap_grad = cv.Laplacian(img, cv.CV_64F, ksize=3)
        lap_grad = np.uint(np.absolute(lap_grad))
        return lap_grad

    def image_contrast(self, img):
        '''
        Turns RGB image to grayscale image and returns the contrast value of the image
        '''
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        min = np.round(np.min(gray), 5)
        max = np.round(np.max(gray), 5)
        if (max + min) == 0:
            return 0
        else:
            contrast = np.round((max - min)/(max+min), 5)
            return contrast

    def LBP(self, img):
        '''
        Calculate Local Binary Pattern
        '''
        def get_pixel(img, center, x, y):
      
            new_value = 0
            
            try:
                # If local neighbourhood pixel 
                # value is greater than or equal
                # to center pixel values then 
                # set it to 1
                if img[x][y] >= center:
                    new_value = 1
                    
            except:
                # Exception is required when 
                # neighbourhood value of a center
                # pixel value is null i.e. values
                # present at boundaries.
                pass
            
            return new_value

        def lbp_calculated_pixel(img, x, y):
   
            center = img[x][y]
        
            val_ar = []
            
            # top_left
            val_ar.append(get_pixel(img, center, x-1, y-1))
            
            # top
            val_ar.append(get_pixel(img, center, x-1, y))
            
            # top_right
            val_ar.append(get_pixel(img, center, x-1, y + 1))
            
            # right
            val_ar.append(get_pixel(img, center, x, y + 1))
            
            # bottom_right
            val_ar.append(get_pixel(img, center, x + 1, y + 1))
            
            # bottom
            val_ar.append(get_pixel(img, center, x + 1, y))
            
            # bottom_left
            val_ar.append(get_pixel(img, center, x + 1, y-1))
            
            # left
            val_ar.append(get_pixel(img, center, x, y-1))
            
            # Now, we need to convert binary
            # values to decimal
            power_val = [1, 2, 4, 8, 16, 32, 64, 128]
        
            val = 0
            
            for i in range(len(val_ar)):
                val += val_ar[i] * power_val[i]
                
            return val
        
        height, width, _ = img.shape
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_lbp = np.zeros((height, width),np.uint8)

        for i in range(0, height):
            for j in range(0, width):
                img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
            
        return img_lbp

    def DCP(self, img, patch_x=12, patch_y=9):
        '''
        Dark Channel Prior algorithm 
        '''

        Y, X, _ = img.shape
        y_pad = int(patch_y/2)
        x_pad = int(patch_x/2)
        padded = np.pad(img, ((y_pad, y_pad), (x_pad, x_pad), (0, 0)), 'edge')
        darkch = np.zeros((Y, X))
        for i, j in np.ndindex(darkch.shape):
            darkch[i, j] = np.min(padded[i:i + patch_y, j:j + patch_x, :])
        return darkch

    def combine_feature(self, img):
        h,s,v = self.hsv(img)
        gradient = self.gradient_magnitude(img)
        lbp = self.LBP(img)
        c = self.image_contrast(img)
        dcp = self.DCP(img)
        return np.concatenate((h.ravel(), s.ravel(), v.ravel(), gradient.ravel(), lbp.ravel(), np.array([c]), dcp.ravel()))
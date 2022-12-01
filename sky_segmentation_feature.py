import cv2 as cv
import numpy as np
import pandas as pd
from Meanshift_image import meanshift

class sky_segmentation:
    def __init(self):
        pass

    def image_rgb(self,img_path):
        '''
        Reads and returns RGB image from the path
        '''
        img = cv.imread(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        return img

    def DCP(self, img, patch_x=12, patch_y=9):
        '''
        Dark Channel Prior algorithm to classify a hazy image. Images are of size 1920 x 1080.
        '''
        Y, X, _ = img.shape
        y_pad = int(patch_y/2)
        x_pad = int(patch_x/2)
        padded = np.pad(img, ((y_pad, y_pad), (x_pad, x_pad), (0, 0)), 'edge')
        darkch = np.zeros((Y, X))
        for i, j in np.ndindex(darkch.shape):
            darkch[i, j] = np.min(padded[i:i + patch_y, j:j + patch_x, :])
        return darkch

    def atmospheric_light(self, img, p=0.1, LAMBDA = 17, DELTA=36):
        # Finding highlighted area
        dcp = self.DCP(img)
        img = img.astype('int')
        r = img[:,:,0]
        g = img[:,:,1]
        b = img[:,:,2]
        m = (r + g + b)/3
        s = LAMBDA * np.sqrt((np.power(r - m,2) + np.power(g-m,2) + np.power(b-m,2)))
        flats = s.ravel()
        highlighted = np.where(flats < DELTA)

        # Exclude highlighted area from DCP
        flat_dcp = dcp.ravel()
        Y,X = dcp.shape
        flat_img=img.reshape(Y*X, 3)
        filtered_flat_dcp = np.copy(flat_dcp)
        np.put(filtered_flat_dcp, highlighted, 0)

        # Find p% highest intensity pixel in DCP
        # searchidx = (-flat_dcp).argsort()[:int(X*Y*p/100)]
        searchidx = (-filtered_flat_dcp).argsort()[:int(X*Y*p/100)]

        # Find those pixels in the original image:
        top_intensity_img = flat_img.take(searchidx, axis=0)
        light_location = np.array([(i % X, i // X) for i in searchidx]) # (x,y)

        # return the mean of those top pixels:
        return light_location.astype('int'),np.mean(top_intensity_img, axis = 0)

    def transmission(self, img, omega = 0.95, t0=0.1):
        '''
        Returns the transmission values of an image

        F: 1 - w DCP(I/A)
        '''
        
        _,A = self.atmospheric_light(img)
        img = img.astype('int')
        t_rate = np.where(img/A > 1, 1, img/A)
        t = 1 - omega * self.DCP(t_rate)
        t = np.where(t < t0, t0, t)
        Y, X = t.shape
        return t.reshape(Y, X, 1)

    def luminance(self,segment):
        '''
        Returns the mean luminance value of a segment (2 dimensions)
        '''
        r = np.mean(segment[:, 0])
        g = np.mean(segment[:, 1])
        b = np.mean(segment[:, 2])
        
        return np.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))

    def gradient(self,img):
        lap_grad = cv.Laplacian(img, cv.CV_64F, ksize=3)
        lap_grad = np.uint(np.absolute(lap_grad))
        return lap_grad


    def combine_feature_vector(self, img):
        '''
        Takes imgX
        Calculate vectors of features for imgX: max transmission, std transmission, mean transmission, mean red, mean green, mean blue, combine mean rgb, median of mean rgb, mean intensity, mean gradient, lightness, mean pos x, mean pos y.
        Add coordinate of segments in the image
        '''
        
        imgX = img.astype('float')
        transmission = self.transmission(imgX)
        gradient = self.gradient(imgX)

        ms = meanshift()

        mnshft = ms.meanshifted_img(imgX)

        img_label = np.reshape(mnshft.labels_, imgX.shape[:2])
        labels = np.unique(mnshft.labels_)
        df = pd.DataFrame(columns=['max t', 'std t', 'mean t', 'mean red', 'mean green', 'mean blue', 'combine mean rgb', 'median of mean rgb', 'mean intensity', 'mean gradient', 'lightness', 'meanX', 'meanY'])
        posX = []
        posY = []
        for label in labels:
            segment = np.where(img_label == label)
            image_segment = imgX[segment[0], segment[1]]

            lightness = self.luminance(image_segment)
            max_t = np.max(transmission[segment[0], segment[1]])
            std_t = np.std(transmission[segment[0], segment[1]])
            mean_t = np.mean(transmission[segment[0], segment[1]])
            mean_r = np.mean(imgX[:,:,0][segment[0], segment[1]])
            mean_g = np.mean(imgX[:,:,1][segment[0], segment[1]])
            mean_b = np.mean(imgX[:,:,2][segment[0], segment[1]])
            mean_rgb = np.mean([mean_r, mean_g, mean_b])
            med_rgb = np.median([mean_r, mean_g, mean_b])
            mean_intensity = np.mean(imgX[:,:,0][segment[0], segment[1]] + imgX[:,:,1][segment[0], segment[1]] + imgX[:,:,2][segment[0], segment[1]])
            mean_grad = np.mean(gradient[segment[0], segment[1]])

            df = df.append({'max t':max_t, 'std t':std_t, 'mean t':mean_t, 'mean red':mean_r, 'mean green':mean_g, 'mean blue':mean_b, 'combine mean rgb':mean_rgb, 'median of mean rgb':med_rgb, 'mean intensity':mean_intensity, 'mean gradient':mean_grad, 'lightness':lightness, 'meanX':np.mean(segment[1]), 'meanY':np.mean(segment[0])}, ignore_index = True)
            posY.append(segment[0])
            posX.append(segment[1])
            
        return df, posY, posX
            
    
    def combine_feature_dataframe(self, img, imgY):
        '''
        Takes imgX, imgY. 
        Calculate vectors of features for imgX: max transmission, std transmission, mean transmission, mean red, mean green, mean blue, combine mean rgb, median of mean rgb, mean intensity, mean gradient, lightness.
        Add also the coordinates in the image'''

        df, posY, posX = self.combine_feature_vector(img)
        yv = []
        for i in range(len(posY)):
            sY = posY[i]
            sX = posX[i]

            label_segment = imgY[sY, sX]
            true_rate = np.sum(label_segment == 1) / label_segment.shape[0]
            if true_rate >= 0.5:
                y = 1
            else:
                y = 0
            yv.append(y)
            
        df['y'] = yv
        return df

    
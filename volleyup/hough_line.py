#!/usr/bin/env python
""" Trying to get court lines """
import cv2
import numpy as np
import math
import config
from utils import get_channel, get_jpgs, get_courtmask
from feature import FeatureDescriptor
import matplotlib.pyplot as plt

class TranslationStitcher():
    """ Assumes only translation offset between images and credited to
        https://github.com/marcpare/stitch/blob/master/crichardt/stitch.py
    """
    def __init__(self, imgs):
        self.ft = FeatureDescriptor()
        self.imgs = imgs

    def generate_mosaic(self, mask_func, channel='hsv_s', feature='akaze'):
        """
        Skeleton from panorama.py     
        OBJECT_COLOR = {'upper_court': [(20, 0, 100), (50, 100, 255)]}
        OBJECT_COLOR = {'nomask': [(0, 0, 0), (255, 255, 255)]}
        """
        
        panorama_img = self.imgs[0]
        prev_img = panorama_img.copy()
        ii = 0
        for index, next_img in enumerate(self.imgs[1:]):
            ii += 1
            img = panorama_img

            ''' roi mask to keep only court + players '''
            img_width, img_height, img_channel = img.shape
            x = 0
            y = 150
            w = img_height - x
            h = img_width - y
            masked_img = np.zeros(img.shape,np.uint8)
            masked_img[y:y+h,x:x+w] = img[y:y+h,x:x+w]
            
            # img = cv2.GaussianBlur(img,(3,3),0)
            gray = cv2.cvtColor(masked_img,cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray,15,100,apertureSize = 3)
            ''' canny edge parameter test'''            
            
            plt.subplot(121),plt.imshow(masked_img[...,::-1],cmap = 'gray')
            plt.title('Masked Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(122),plt.imshow(edges,cmap = 'gray')
            plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

            #plt.show()
            ''' canny edge detector displayed above '''

            minLineLength = 23
            #minLineLength = img.shape[1]-300
            maxLineGap = 10
            #lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
            #lines = cv2.HoughLines(edges,1,np.pi/180,0)
            #lines = cv2.HoughLines(edges,0.01,np.pi/720,100)
            lines = cv2.HoughLinesP(image=edges,rho=0.02,theta=np.pi/360,
                                    threshold=1,lines=np.array([]),
                                    minLineLength=minLineLength,maxLineGap=maxLineGap)

            a,b,c = lines.shape
            for i in range(a):
                cv2.line(img, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)

            cv2.imwrite('houghlines'+str(ii)+'.jpg',img)
            plt.show()
        return panorama_img

      
imgs = get_jpgs(config.INDVIDUAL_VIDEOS['3'], skip=10)
stitcher = TranslationStitcher(imgs)
stitcher.generate_mosaic(get_courtmask)
cv2.destroyAllWindows()

#!/usr/bin/env python
""" Performs image stitching on images provided """
import cv2
import numpy as np
import math
import config
import os
from utils import get_channel, get_jpgs, get_netmask
from feature import FeatureDescriptor


class TranslationStitcher():
    """ Assumes only translation offset between images and credited to
        https://github.com/marcpare/stitch/blob/master/crichardt/stitch.py
    """
    def __init__(self, imgs):
        self.ft = FeatureDescriptor()
        self.imgs = imgs
        
    def hough_line(self,current_img):

        # img = cv2.GaussianBlur(img,(3,3),0)
        gray = cv2.cvtColor(current_img,cv2.COLOR_BGR2GRAY)
        #threshold = [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_TRUNC, cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV ]
        #ret,masked_img = cv2.threshold(masked_img,127,255,cv2.THRESH_TOZERO_INV)
        edges = cv2.Canny(gray,15,100,apertureSize = 3)
        ''' canny edge parameter test'''            

        ''' roi mask to keep only court + players '''
        img_width, img_height, img_channel = current_img.shape
        x = 0
        y = 150
        w = img_height - x - 100
        h = img_width - y
        masked_img = np.zeros(edges.shape,np.uint8)
        masked_img[y:y+h,x:x+w] = edges[y:y+h,x:x+w]
        edges = masked_img
        #plt.subplot(121),plt.imshow(masked_img[...,::-1],cmap = 'gray')
        #plt.title('Masked Image'), plt.xticks([]), plt.yticks([])
        #plt.subplot(122),plt.imshow(edges,cmap = 'gray')
        #plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        
        #plt.show()
        ''' canny edge detector displayed above '''
        
        minLineLength = 23
        #minLineLength = img.shape[1]-300
        maxLineGap = 9
        #lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
        #lines = cv2.HoughLines(edges,1,np.pi/180,0)
        #lines = cv2.HoughLines(edges,0.01,np.pi/720,100)
        isProbabilistic = True
        if not isProbabilistic:
            lines = cv2.HoughLines(edges,0.04,np.pi/360,100)
            for rho,theta in lines[0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                cv2.line(current_img,(x1,y1),(x2,y2),(0,0,255),2)
        if isProbabilistic:
            lines = cv2.HoughLinesP(image=edges,rho=0.04,theta=np.pi/360,
                                threshold=1,
                                minLineLength=23,maxLineGap=9)
            a,b,c = lines.shape
            lineNum = 0
            for i in range(a):
                cv2.line(current_img, (lines[i][0][0], lines[i][0][1]),
                         (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
                lineNum=lineNum+1
        #cv2.imshow('new',current_img)
        #cv2.waitKey(0)
        return current_img
    
    def calc_matches(self, desc1, desc2, method='bf'):
        """ Calculate matches between descriptors specified by given method
        Parameters
        ----------
        method : bf    (brute force matching)
                 flann (fast nearest neighbor matching)

        """
        if method is 'bf':
            bf = cv2.BFMatcher()
            return bf.knnMatch(desc1, desc2, k=2)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        return flann.knnMatch(desc1, desc2, k=2)

    def match_features(self, desc1, desc2, ratio=0.8):
        """ Matches features and filter only good matches using Lowe's ratio """
        matches = self.calc_matches(desc1, desc2)
        good = [m for m, n in matches if m.distance < ratio * n.distance]
        return good

    def calc_translation(self, src_pts, dst_pts):
        m = cv2.estimateRigidTransform(src_pts, dst_pts, True)
        return np.float32([[1, 0, m[0, 2]],
                           [0, 1, 0]])

    def calc_homography_affine(self, imgA, imgB, kp1, kp2, good_matches,
                               min_good_match=4, reproj_thresh=4.0):
        """ Calculates homography and affine transformation when there is at least 8 matches
        Parameters
        ----------
        min_good_match : minimum number of good matches before calculating homography
        reproj_thresh  : maximum allowed reprojection error for RANSAC to be treated as inlier
        """
        if len(good_matches) >= min_good_match:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_thresh)
            affine = self.calc_translation(dst_pts, src_pts)
            return (H, status, affine)

        print ("Not enough matches are found - %d/%d") % (len(good_matches), min_good_match)
        return None

    def calculate_size(self, imgA_size, imgB_size, homography):
        (h1, w1) = imgA_size[:2]
        (h2, w2) = imgB_size[:2]

        # remap the coordinates of the projected image onto the panorama image space
        top_left = np.dot(homography, np.asarray([0, 0, 1]))
        top_right = np.dot(homography, np.asarray([w2, 0, 1]))
        bottom_left = np.dot(homography, np.asarray([0, h2, 1]))
        bottom_right = np.dot(homography, np.asarray([w2, h2, 1]))

        # normalize
        top_left = top_left/top_left[2]
        top_right = top_right/top_right[2]
        bottom_left = bottom_left/bottom_left[2]
        bottom_right = bottom_right/bottom_right[2]

        pano_left = int(min(top_left[0], bottom_left[0], 0))
        pano_right = int(max(top_right[0], bottom_right[0], w1))
        W = pano_right - pano_left

        pano_top = int(min(top_left[1], top_right[1], 0))
        pano_bottom = int(max(bottom_left[1], bottom_right[1], h1))
        H = pano_bottom - pano_top

        size = (W, H)

        # offset of first image relative to panorama
        X = int(min(top_left[0], bottom_left[0], 0))
        Y = int(min(top_left[1], top_right[1], 0))
        offset = (-X, -Y)
        return (size, offset)

    def merge_images_translation(self, imgA, imgB, offset):
        # Put images side-by-side into 'image'.
        (h1, w1) = imgA.shape[:2]
        (h2, w2) = imgB.shape[:2]
        (ox, oy) = offset
        ox = int(ox)
        oy = int(oy)
        h, w = max(h1, h2), max(w1, w2)
        image = np.zeros((h, w + abs(ox), 3), np.uint8)
        image[:h1, :w1] = imgA
        image[:h2-oy, ox:ox+w2] = imgB[oy:h2, :]
        return image

    def generate_mosaic(self, mask_func, channel='hsv_s', feature='akaze'):
        """ Generates image mosaics from images
        Parameters
        ----------
        mask_func : mask function to produce mask used to reduce search area of feature detection
        channel   : channel used for processing
        feature   : feature detector for interest point detection

        """
        panorama_img = self.imgs[0]
        panorama_img_list = []
        panorama_img_list.append(panorama_img.copy())
        prev_img = panorama_img.copy()
        for index, next_img in enumerate(self.imgs[1:]):
            img1 = panorama_img.copy()
            img2 = next_img.copy()
            img1 = self.hough_line(img1)
            img2 = self.hough_line(img2)
            #cv2.imshow('original | hough', np.hstack((img1, panorama_img)))
            cv2.waitKey(0)
            kp1, desc1 = self.ft.compute(get_channel(img1, channel), feature,
                                         mask_func(img1))
            kp2, desc2 = self.ft.compute(get_channel(img2, channel), feature,
                                         mask_func(img2))
            matches = self.match_features(desc1, desc2)
            (H, status, affine) = self.calc_homography_affine(img1, img2, kp1,
                                                              kp2, matches)
            if H is not None:
                (size, offset) = self.calculate_size(panorama_img.shape, panorama_img.shape, H)
                if index == 100:
                    #self.debug_matching(img1, img2, mask_func, channel, feature)
                    return panorama_img_list
                next_img = cv2.warpPerspective(next_img, H,
                                              (panorama_img.shape[1], panorama_img.shape[0]),
                                              panorama_img, borderMode=cv2.BORDER_TRANSPARENT)
                
                #panorama_img = self.merge_images_translation(panorama_img, next_img, offset)
                #cv2.imshow('panorama', base_img)
                #cv2.waitKey(10)
                print("merged img ",index)
            #panorama_img = next_img.copy()
            prev_img = next_img.copy()
            panorama_img_list.append(panorama_img.copy())
        print("final list len",len(panorama_img_list))
        return panorama_img_list

    def get_avg_translation(self, kp1, kp2, matches):
        mean_x = np.mean([kp1[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0] for m in matches])
        mean_y = np.mean([kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1] for m in matches])
        return int(mean_x), int(mean_y)

    def debug_matching(self, imgA, imgB, mask_func, channel, feature, wait=0):
        """ Displays debugging window for each images along with detected keypoints """
        kp1, desc1 = self.ft.compute(get_channel(imgA, channel), feature,
                                     mask_func(imgA))
        kp2, desc2 = self.ft.compute(get_channel(imgB, channel), feature,
                                     mask_func(imgB))
        matches = self.match_features(desc1, desc2)
        (H, status, affine) = self.calc_homography_affine(imgA, imgB, kp1, kp2, matches)
        cv2.drawKeypoints(imgA, kp1, imgA, config.BLUE, 1)
        cv2.drawKeypoints(imgB, kp2, imgB, config.BLUE, 1)
        warpedA = cv2.warpPerspective(imgA, H, (imgB.shape[1], imgB.shape[0]))
        warpedB = cv2.warpPerspective(imgB, np.linalg.inv(H), (imgA.shape[1], imgA.shape[0]))
        matched = cv2.drawMatches(imgA, kp1, imgB, kp2, matches, None, flags=2)
        cv2.imshow('matched | warped', np.vstack((matched, np.hstack((warpedA, warpedB)))))
        cv2.waitKey(wait)
def convertToVideo(dirpath, segment, type):
    imgs = get_jpgs(dirpath)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter("output_" + type + str(segment) + ".mov", fourcc, 30, (imgs[0].shape[1], imgs[0].shape[0]))
    print("Writing video ...")
    i = 1
    for img in imgs:
        print ("Writing image", i)
        i+=1
        vw.write(img)
    
    vw.release()

def write_jpgs(dirpath, jpgs):
    """ 
    Writes all images to the given dirpath
        
    """
    if os.path.exists(os.path.abspath(dirpath)):
        print ("no of imgs",len(jpgs))
        for i in range(len(jpgs)):
            filename = dirpath + str(i) + ".jpg"
            cv2.imwrite(filename, jpgs[i])
        print('Wrote {} images to {}'.format(len(jpgs), dirpath))
    print('Directory {} does not exist'.format(os.path.abspath(dirpath)))

def get_nomask(img):
    return img
imgs = get_jpgs(config.INDVIDUAL_VIDEOS['3'], skip=0)
stitcher = TranslationStitcher(imgs)
panorama_list = stitcher.generate_mosaic(get_nomask)
number = 3
write_jpgs(config.DATA_DIR + "processedImages" + str(number) + "/", jpgs=panorama_list)
convertToVideo(config.DATA_DIR + "processedImages" + str(number) + "/", number, "akaze")
#cv2.imshow('image',mosaic_img)
#cv2.waitKey(0)
cv2.destroyAllWindows()

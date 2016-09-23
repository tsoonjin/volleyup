#!/usr/bin/env python
""" Performs image stitching on images provided, to create a panorama view """
import cv2
import numpy as np
import math
import config
from utils import get_channel, get_jpgs, get_netmask
from feature import FeatureDescriptor
from cv2 import KeyPoint

class TranslationStitcher():
    """ Assumes only translation offset between images and credited to
        https://github.com/marcpare/stitch/blob/master/crichardt/stitch.py
    """
    def __init__(self, imgs):
        self.ft = FeatureDescriptor()
        self.imgs = imgs

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

    def calc_homography(self, imgA, imgB, src_pts, dst_pts, min_good_match=4, reproj_thresh=4.0):
        """ Calculates homography and affine transformation when there is at least 8 matched feature points (4 in each image)
        Parameters
        ----------
        min_good_match : minimum number of good matches before calculating homography
        reproj_thresh  : maximum allowed reprojection error for RANSAC to be treated as inlier
        """
        #src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        #dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_thresh)
        return (H, status)

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

    def generate_mosaic(self, mask_func, channel='hsv_s'):
        """ Generates image mosaics from images
        Parameters
        ----------
        mask_func : mask function to produce mask used to reduce search area of feature detection
        channel   : channel used for processing

        """
        panorama_img = self.imgs[0]
        imgA = panorama_img.copy()
        for index, imgB in enumerate(self.imgs[1:]):
            gray_imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
            gray_imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
            gray_imgA = np.float32(gray_imgA)
            gray_imgB = np.float32(gray_imgB)
            dst1 = cv2.cornerHarris(gray_imgA, 2, 3, 0.04)
            dst2 = cv2.cornerHarris(gray_imgB, 2, 3, 0.04)
            
            dst1 = cv2.dilate(dst1,None)
            ret, dst1 = cv2.threshold(dst1,0.01*dst1.max(),255,0)
            dst1 = np.uint8(dst1)
            ret1, labels1, stats1, centroids1 = cv2.connectedComponentsWithStats(dst1)
            
            dst2 = cv2.dilate(dst2,None)
            ret, dst2 = cv2.threshold(dst2,0.01*dst2.max(),255,0)
            dst2 = np.uint8(dst2)
            ret2, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(dst2)

            criteria1 = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            corners1 = cv2.cornerSubPix(dst1,np.float32(centroids1),(5,5),(-1,-1),criteria1)
            result1 = np.hstack((centroids1, corners1))
            result1 = np.int0(result1)
            imgA[result1[:,1],result1[:,0]]=[0,0,255]
            imgA[result1[:,3],result1[:,2]] = [0,255,0]
            
            criteria2 = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            corners2 = cv2.cornerSubPix(dst2,np.float32(centroids2),(5,5),(-1,-1),criteria2)
            result2 = np.hstack((centroids2, corners2))
            result2 = np.int0(result2)
            imgB[result2[:,1],result2[:,0]]=[0,0,255]
            imgB[result2[:,3],result2[:,2]] = [0,255,0]
            
            kp1 = []
            mat1 = np.empty([0,0])
            for i in range(len(dst1)):
                for j in range(len(dst1[i])):
                    if dst1[i,j] == 255:
                        kp = KeyPoint(float(i), float(j), 10.0)
                        np.append(mat1, kp)
                        kp1.append(kp)
            
            kp2 = []
            mat2 = np.empty([0,0])
            for i in range(len(dst2)):
                for j in range(len(dst2[i])):
                    if dst2[i,j] == 255:
                        kp = KeyPoint(float(i), float(j), 10.0)
                        np.append(mat2, kp)
                        kp2.append(kp)
            
            
            matches = self.match_features(mat1, mat2)
            matched = cv2.drawMatches(imgA, kp1, imgB, kp2, matches, None, flags=2)
            print "Matches: ", matches
            cv2.imshow('matched | warped', matched)
            cv2.waitKey(0)

            panorama_img = imgB.copy()
            imgA = imgB.copy()
        return panorama_img

if __name__ == '__main__':
    # Put extracted images into DATA_DIR before running this
    imgs = get_jpgs(config.DATA_DIR + "beachVolleyball2/")
    cv2.ocl.setUseOpenCL(False)
    stitcher = TranslationStitcher(imgs)
    stitcher.generate_mosaic(get_netmask)
    cv2.destroyAllWindows()

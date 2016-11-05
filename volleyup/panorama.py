#!/usr/bin/env python
""" Performs image stitching on images provided, to create a panorama view """
import cv2
import numpy as np
import math
import config
import os
from utils import get_channel, get_jpgs, get_netmask, write_jpgs
from feature import FeatureDescriptor

class TranslationStitcher():
    def __init__(self, imgs):
        self.ft = FeatureDescriptor()
        self.imgs = imgs
    
    def calc_matches(self, desc1, desc2):
        bf = cv2.BFMatcher()
        return bf.knnMatch(desc1, desc2, 2)
    
    def match_features(self, desc1, desc2, ratio=0.3):
        matches = self.calc_matches(desc1, desc2)
        good_matches = [m for m, n in matches if m.distance < ratio * n.distance]
        return good_matches
    
    def calc_homography(self, imgA, imgB, kp1, kp2, good_matches, min_good_match=4, reproj_thresh=5.0):
        """
            Calculates homography when there are at least 8 matched feature points (4 in each image)
            Parameters
            ----------
            min_good_match : minimum number of good matches before calculating homography
            reproj_thresh  : maximum allowed reprojection error for RANSAC to be treated as inlier
            """
        if len(good_matches) >= min_good_match:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_thresh)
            return (H, status)
        
        print "Not enough matches are found - %d/%d" % (len(good_matches), min_good_match)
        return None
    
    def generate_panorama(self, mask_func, channel='hsv_s', feature='akaze'):
        panorama_img_list = []
        """ Generates image panorama
            Parameters
            ----------
            mask_func : mask function to produce mask used to reduce search area of feature detection (currently, this is not used)
            channel   : channel used for processing
            feature   : feature detector for interest point detection
            """
        sift = cv2.xfeatures2d.SIFT_create()# testing with sift instead of akaze
        panorama_img = self.imgs[0]
        resultingHomography = None
        panorama_img = np.pad(panorama_img, ((200,200),(800,800),(0,0)), mode='constant')
        imgA = panorama_img.copy()
        panorama_img_list.append(panorama_img.copy())
        for index, imgB in enumerate(self.imgs[1:]):
            print "Processing image", index
            masked_imgA = cv2.cvtColor(imgA, cv2.COLOR_RGB2GRAY)
            masked_imgB = cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY)
            key_points_A, desc1 = self.ft.compute(imgA, feature, masked_imgA)
            key_points_B, desc2 = self.ft.compute(imgB, feature, masked_imgB)
            #key_points_A, desc1 = sift.detectAndCompute(masked_imgA, None)
            #key_points_B, desc2 = sift.detectAndCompute(masked_imgB, None)
            # Match feature descriptors and filter which keeps the good ones
            matching_features = self.match_features(desc2, desc1)
            
            #matched = cv2.drawMatches(imgB, key_points_B, imgA, key_points_A, matching_features, None, flags=2)
            #cv2.imshow('Matched', matched)
            #cv2.waitKey(10)
            
            # Calculate the homography matrix and affine required to transform imgB to imgA (so that the matching points overlap)
            if len(matching_features) >= 4:
                (H, status) = self.calc_homography(imgB, imgA, key_points_B, key_points_A, matching_features)
            else:
                print "Not enough matching features"
            if H is not None:
                if resultingHomography is None:
                    resultingHomography = np.matrix(H)
                else:
                    resultingHomography = resultingHomography * np.matrix(H)
                
                warpedB = imgB.copy()
                warpedB = cv2.warpPerspective(warpedB, resultingHomography, (panorama_img.shape[1], panorama_img.shape[0]), panorama_img, borderMode=cv2.BORDER_TRANSPARENT)
                
                panorama_img_list.append(panorama_img.copy())
            imgA = imgB.copy()
        
        return panorama_img_list

def convertToVideo(dirpath, segment, type):
    imgs = get_jpgs(dirpath)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter("output_" + type + str(segment) + ".mov", fourcc, 30, (imgs[0].shape[1], imgs[0].shape[0]))
    print "VideoWriter is opened:", vw.isOpened()
    print("Writing video ...")
    i = 1
    for img in imgs:
        print "Writing image", i
        i+=1
        vw.write(img)
    
    vw.release()

def stitch(number):
    ## Put extracted images into DATA_DIR/<folder> before running this
    imgs = get_jpgs(config.DATA_DIR + "beachVolleyball" + str(number) + "/")
    cv2.ocl.setUseOpenCL(False) # A workaround for ORB feature detector error
    stitcher = TranslationStitcher(imgs)
    panorama_list = stitcher.generate_panorama(get_netmask)
    
    # Create the folder
    d = os.path.dirname(config.DATA_DIR + "processedImages" + str(number) + "/")
    if not os.path.exists(d):
        os.makedirs(d)
    else:
        filelist = os.listdir(d)
        for file in filelist:
            os.remove(d + "/" + file)
    
    write_jpgs(config.DATA_DIR + "processedImages" + str(number) + "/", jpgs=panorama_list)
    convertToVideo(config.DATA_DIR + "processedImages" + str(number) + "/", number, "akaze")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    for i in range(1,8):
        stitch(i)

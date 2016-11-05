#!/usr/bin/env python
""" Performs image stitching on images provided, to create a panorama view """
import cv2
import numpy as np
import math
import config
import os
from utils import get_channel, get_jpgs, get_netmask, write_jpgs
from feature import FeatureDescriptor
from collections import Counter

def imfill(im):
    h, w, _ = im.shape
    im_floodfill = im.copy()
    mask = np.zeros((h +2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), (255, 255, 255))
    
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    return im | im_floodfill_inv

def copyOver(source, destination):
    result_grey = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(result_grey, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    roi = cv2.bitwise_and(source, source, mask=mask)
    im2 = cv2.bitwise_and(destination, destination, mask=mask_inv)
    result = cv2.add(im2, roi)
    return result

class CourtFinder:
    def process_frame(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hues = frame[:,:,0]
        maximal_peak = max(Counter(hues.ravel()).items(), key = lambda x: x[1])[0]
        rotation = int(90 - maximal_peak)
        rotated = cv2.add(np.int8(hsv), rotation) % 180
        mask = cv2.inRange(rotated, np.array([60, 0, 0]), np.array([120, 255, 255]))
        im, mask_contours, hierachy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#, key = cv2.contourArea, reverse = True
        mask_contours = sorted(mask_contours, key = cv2.contourArea, reverse = True)[:10]
        
        mc_image = np.zeros_like(frame)
        mc_image = cv2.drawContours(mc_image, mask_contours, -1, [255, 255, 255], cv2.FILLED)
        # returns the mask
        return cv2.cvtColor( imfill(mc_image), cv2.COLOR_BGR2GRAY)


class TranslationStitcher():
    """ Assumes only translation offset between images and credited to
        https://github.com/marcpare/stitch/blob/master/crichardt/stitch.py
    """
    def __init__(self, imgs):
        self.ft = FeatureDescriptor()
        self.imgs = imgs

    def calc_matches(self, desc1, desc2, method='flann'):
        """ Calculate matches between descriptors specified by given method
        Parameters
        ----------
        method : bf    (brute force matching)

        """
        """ Alternatively?
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(desc1, desc2, 2)
        return rawMatches
        """
        bf = cv2.BFMatcher()
        return bf.knnMatch(desc1, desc2, 2)

    def match_features(self, desc1, desc2, ratio=0.8):
        """ Matches features and filter only good matches using Lowe's ratio """
        matches = self.calc_matches(desc1, desc2)
        good_matches = [m for m, n in matches if m.distance < ratio * n.distance]
        return good_matches
    
    def calc_translation(self, src_pts, dst_pts):
        m = cv2.estimateRigidTransform(src_pts, dst_pts, True)
        return np.float32([[1, 0, m[0, 2]],
                           [0, 1, 0]])

    def calc_homography(self, imgA, imgB, kp1, kp2, good_matches, min_good_match=4, reproj_thresh=3.0):
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

    def calc_affine(self, src_pts, dst_pts):
        """
            Calculates affine transformation required for destination points of the destination image to match the source points
        """
        affine = self.calc_translation(dst_pts, src_pts)
        return affine

    def overlay_image(self, mainImage, overlayImage):
        """
           Overlays overlayImage onto mainImage, assuming that important parts of overlayImage are not totally black i.e. (0,0,0)
        """
        return np.where(overlayImage < [30,30,30], mainImage, overlayImage)

    def generate_panorama(self, mask_func, channel='hsv_s', feature='akaze'):
        panorama_img_list = []
        params = dict(maxCorners=500, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        homographyStack = []
        cf = CourtFinder()
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
        #mask =  255 - cf.process_frame(panorama_img)
        #cv2.imshow('MASK', mask)
        #cv2.waitKey(10)
        # Add in margins in all directions
        #panorama_img = np.pad(panorama_img, ((100,0),(0,100),(0,0)), mode='constant')
        imgA = panorama_img.copy()
        
        panorama_img_list.append(panorama_img.copy())
        #imgA = panorama_img.copy()
        for index, imgB in enumerate(self.imgs[1:]):
            print "Processing image", index+2
        
            mask =  255 - cf.process_frame(imgB)
            cv2.imshow('MASK', mask)
            k = cv2.waitKey(10)
            if k == 27:
                print("Terminated by user")
                exit()
            #imgB = np.pad(imgB, ((100,100),(100,100),(0,0)), mode='constant')
            
            #mask =  255 - cf.process_frame(imgA)
            #cv2.imshow('MASK', mask)
            #k = cv2.waitKey(10)
            #if k == 27:
            #    print("Terminated by user")
            #    exit()
            grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
            
            goodFeatures = cv2.goodFeaturesToTrack(imgA, mask=mask, **params)
            if goodFeatures is None:
                print "Error, no good features to track."
                continue
            
            p1, st, err = cv2.calcOpticalFlowPyrLK(grayA, grayB, goodFeatures, None, **self.lk_params)
            p0r, st, err = cv2.calcOpticalFlowPyrLK(grayB, grayA, p1, None, **self.lk_params)

            points_A = []
            points_B = []
            tracks = []
            d = abs(goodFeatures-p0r).reshape(-1, 2).max(-1)
            good = d < 1
            for (xA, yA), (xB, yB), good_flag in zip(goodFeatures.reshape(-1, 2), p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                points_A.append((xA, yA))
                points_B.append((xB, yB))

            H, status = cv2.findHomography(np.int32(points_B), np.int32(points_A), cv2.RANSAC, 5.0)
            if H is not None:
                homographyStack.append(H)
                warpedB = imgB.copy()
                #warpedB = np.pad(warpedB, ((100,0),(0,100),(0,0)), mode='constant')
                for ho in reversed(homographyStack):
                    warpedB = cv2.warpPerspective(warpedB, ho, (panorama_img.shape[1], panorama_img.shape[0]))
                #panorama_img = self.overlay_image(panorama_img, warpedB)
                panorama_img = copyOver(warpedB, panorama_img)
                
                panorama_img_list.append(panorama_img.copy())
            else:
                print "Warning, homography cannot be found"
            imgA = imgB.copy()

        return panorama_img_list

def convertToVideo(dirpath):
    imgs = get_jpgs(dirpath)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter("output.mov", fourcc, 30, (imgs[0].shape[1], imgs[0].shape[0]))#(imgs[0].shape[1]+400, imgs[0].shape[0]+400))
    print "VideoWriter is opened:", vw.isOpened()
    print("Writing video ...")
    i = 1
    for img in imgs:
        print "Writing image", i
        i+=1
        vw.write(img)

    vw.release()


if __name__ == '__main__':
    number = 3 # Change this number to perform the stitch on different segments
    ## Put extracted images into DATA_DIR/<folder> before running this
    imgs = get_jpgs(config.DATA_DIR + "beachVolleyball" + str(number) + "/")
    cv2.ocl.setUseOpenCL(False) # A workaround for ORB feature detector error
    stitcher = TranslationStitcher(imgs[::5])
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
    convertToVideo(config.DATA_DIR + "processedImages" + str(number) + "/")
    cv2.destroyAllWindows()

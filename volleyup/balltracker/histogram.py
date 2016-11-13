import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def imfill(im):
    h, w, _ = im.shape
    im_floodfill = im.copy()
    mask = np.zeros((h +2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), (255, 255, 255))
    
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    return im | im_floodfill_inv



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


# print(frame.shape)
# print(maximal_peak, rotation)
# print(rotation)
# print(hsv[295][25])
# rotated = cv2.add(np.int8(hsv), rotation) % 180
# print(rotated[295][25])
# mask = cv2.inRange(rotated, np.array([60, 0, 0]), np.array([120, 255, 255]))
# im, mask_contours, hierachy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#, key = cv2.contourArea, reverse = True
# mask_contours = sorted(mask_contours, key = cv2.contourArea, reverse = True)[:10]

# mc_image = np.zeros_like(frame)
# mc_image = cv2.drawContours(mc_image, mask_contours, -1, [255, 255, 255], cv2.FILLED)

cr = CourtFinder()
if __name__ == "__main__":
    while True:
        # cv2.imshow("original", frame)
        # cv2.imshow("mask", mask)
        # cv2.imshow("mc_image", mc_image)
        cv2.imshow("cr", cr.process_frame(frame))
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

video_path = "beachVolleyball\\beachVolleyball1.mov"
cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
hues = frame[:,:,0]
maximal_peak = max(Counter(hues.ravel()).items(), key = lambda x: x[1])[0]
rotation = int(90 - maximal_peak)
filter_width = 15

print(frame.shape)
print(maximal_peak, rotation)
print(rotation)
print(hsv[295][25])
rotated = cv2.add(np.int8(hsv), rotation) % 180
print(rotated[295][25])
mask = cv2.inRange(rotated, np.array([60, 0, 0]), np.array([120, 255, 255]))
im, mask_contours, hierachy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#, key = cv2.contourArea, reverse = True
mask_contours = sorted(mask_contours, key = cv2.contourArea, reverse = True)[:10]

mc_image = np.zeros_like(frame)
mc_image = cv2.drawContours(mc_image, mask_contours, -1, [255, 255, 255], cv2.FILLED)

canny = cv2.Canny(mc_image, 50, 100, apertureSize = 3)

lines = cv2.HoughLines(canny, 1, np.pi/180, 80)
line_image = np.zeros_like(frame)
print(lines[0])
for ((rho, theta),) in lines:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),2)

while True:
    
    cv2.imshow("original", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("mc_image", mc_image)
    cv2.imshow("canny", canny)
    cv2.imshow("line_image", line_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

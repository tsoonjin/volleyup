#things to do:
#determine the 4 corners in the input frame  --to be done after receive input
#read position frame by frame

# import the necessary packages
from __future__ import print_function
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import matplotlib.pyplot as plt






def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect
def getMatrixOfFrame(i,cornerdata):
        c1x = cornerdata[i][1]
        c1y = cornerdata[i][2]
        c2x = cornerdata[i][3]
        c2y = cornerdata[i][4]
        c3x = cornerdata[i][5]
        c3y = cornerdata[i][6]
        c4x = cornerdata[i][7]
        c4y = cornerdata[i][8]

        pts=np.zeros((4, 2), dtype = "float32")
        pts[0][0]=(c1x)
        pts[0][1]=(c1y)
        
        pts[1][0]=(c2x)
        pts[1][1]=(c2y)
        
        pts[2][0]=(c3x)
        pts[2][1]=(c3y)
        
        pts[3][0]=(c4x)
        pts[3][1]=(c4y)        
        #arg = "".join([begin,c1x,comma,c1y,middle,c2x,comma,c2y,middle,c3x,comma,c3y,middle,c4x,comma,c4y,ending])
        #args["coords"] = arg
        #print("corner",arg)

        #pts = np.array(eval(args["coords"]), dtype = "float32")
        


        source2 = np.array([
                [330, 76],
                [700, 200],
                [0, 600],
                [42, 91]], dtype = "float32")

        source1 = np.array([[ 230.,   39.],
               [ 630.,  124.],
               [ 200.,  -200.],
               [  0.,   70.]], dtype="float32")
        
        dst = np.array([
                [0, 0],
                [600-1, 0],
                [600-1, 300-1],
                [0, 300-1]], dtype = "float32")
        

        #print ('rect',rect)
        if i <2 :
                print ('source is ',source1)
                M = cv2.getPerspectiveTransform(source1, dst)
        if i >1:
                print ('source is ',source2)
                M = cv2.getPerspectiveTransform(source2, dst)

        
        print ('dest is ',dst)
        # compute the perspective transform matrix and then apply it
        #M = cv2.getPerspectiveTransform(order_points(pts), dst)
        #M = four_point_transform(pts)
        return M
def four_point_transform(pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	dst = np.array([
		[0, 0],
		[600, 0],
		[600, 300],
		[0, 300]], dtype = "float32")
	print ('dest is ',dst)

	print ('rect',rect)

	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	print ("M is",M)
	return M



# construct the argument parse and parse the arguments




ap = argparse.ArgumentParser()
ap.add_argument("-c", "--coords",help = "comma seperated list of source points")
args = vars(ap.parse_args())
#args["coords"] = "[(205 , 221), (545, 228), (541, 345), (143, 333)]"
args["coords"] = "[(0 , 0), (632, 0), (632, 300), (0, 300)]"
#args["coords"] = "[(200 , 0), (600, 100), (300, 300), (0, 150)]"
#args["coords"] = "[(224,37),(632,126),(200,100,),(-50,75)]"
pts = np.array(eval(args["coords"]), dtype = "float32")
print("original",pts)
#M = four_point_transform(pts)
#print("original M",M)

#dest = np.array(((-50, -50), (50, -50), (-50, 50), (50, 50)), dtype=np.float32)
#M = cv2.getPerspectiveTransform(pts, dest)

# initialize the video stream and allow the camera
print("starting")
#vs = VideoStream(usePiCamera=-1 > 0).start()
#time.sleep(2.0)

# initialize the FourCC, video writer, dimensions of the frame, and
# zeros array
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
writer = None
(h, w) = (None, None)
zeros = None

frameNo = 6000

#read player and ball movements, now just simulate
reader = open('movement.txt')
x = 100
y = 100

ball = open("ball_v1.txt")
balldata  = np.genfromtxt(ball, delimiter=",")


corner = open('corner1.txt')
cornerdata  = np.genfromtxt(corner, delimiter=",")
ball.close()
corner.close()

print("balldata",balldata)
M = getMatrixOfFrame(0,cornerdata)
#for testing######
frameNo = 380
x = balldata[0][1]
y = balldata[0][2]
# for calibrating corners
imagetest = cv2.imread("data/beachVolleyball5/1172.jpg")
#imagetest = cv2.imread("volley_field.jpg")
imagetest = imagetest[...,::-1]
plt.figure()
plt.imshow(imagetest,cmap='gray')
plt.hold(True)
print("shape1",imagetest.shape)
print("shape2",imagetest[0].shape)
plt.scatter(1,2,color='blue')
plt.show()


#############

#M =getMatrixOfFrame(0,cornerdata)
#print("M",M)
#for i in range(frameNo):

                
	

	
	

# do a bit of cleanup
print("[INFO] cleaning up...")
#cv2.destroyAllWindows()
#vs.stop()
writer.release()

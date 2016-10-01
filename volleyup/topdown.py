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
	print ('dest is ',dst)

	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	print ("M is",M)
	return M



# construct the argument parse and parse the arguments




ap = argparse.ArgumentParser()
ap.add_argument("-c", "--coords",help = "comma seperated list of source points")
args = vars(ap.parse_args())
args["coords"] = "[(205 , 221), (545, 228), (541, 345), (143, 333)]"
#args["coords"] = "[(0 , 0), (1300, 0), (1300, 750), (0, 750)]"
pts = np.array(eval(args["coords"]), dtype = "float32")
M = four_point_transform(pts)


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

for i in range(frameNo):
	# grab the frame from the video stream and resize it to have a
	# maximum width of 300 pixels
	#frame = vs.read()
	frame = cv2.imread('volley_field.jpg')
	#frame = cv2.imread('0.jpg')
	frame = imutils.resize(frame, width=600)

	# check if the writer is None
	if writer is None:
		# store the image dimensions, initialzie the video writer,
		# and construct the zeros array
		(h, w) = frame.shape[:2]
		writer = cv2.VideoWriter("example.avi", fourcc, 10,
			(w , h ), True)
		zeros = np.zeros((h, w), dtype="uint8")

	#build content(random data $ color, demo only)	
	line = reader.readline().split(',') #now is not working
	point = (int(line[1]),int(line[2]))
	color = (0, 0, 0)
	
	x = x+1
	y = y+1		
	point = (x,y)
	#nPoint = point
	
	
	point2 = (x+100, y+100)

	#original = np.array([((42, 42), (30, 100))], dtype=np.float32)
	original = np.array([(point, point2)], dtype=np.float)
	converted = cv2.perspectiveTransform(original, M)
	print ("converted",converted)
	print ("converted 0",converted[0][0].item(0))
	print ("converted 1",converted[0][1].item(1))

	npoint1 = (int(converted[0][0].item(0)),int(converted[0][0].item(1)))
	npoint2 = (int(converted[0][1].item(0)),int(converted[0][1].item(1)))
	print('initial point1 ',point)
	print('after transformed point1  ',npoint1)
	print('initial point2 ',point2)
	print('after transformed point2 ',npoint1)
	cv2.circle(frame, npoint1, 10, color, -1)
	cv2.circle(frame, npoint2, 10, color, -1)
	
		
	writer.write(frame)

	# show the frames
	cv2.imshow("Frame", frame)
	cv2.imshow("Output", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
#vs.stop()
writer.release()

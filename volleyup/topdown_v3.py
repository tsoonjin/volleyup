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
        half = cornerdata[i][1]
        c1x = cornerdata[i][2]
        c1y = cornerdata[i][3]
        c2x = cornerdata[i][4]
        c2y = cornerdata[i][5]
        c3x = cornerdata[i][6]
        c3y = cornerdata[i][7]
        c4x = cornerdata[i][8]
        c4y = cornerdata[i][9]

        pts=np.zeros((4, 2), dtype = "float32")
        pts[0][0]=(c1x)
        pts[0][1]=(c1y)
        
        pts[1][0]=(c2x)
        pts[1][1]=(c2y)
        
        pts[2][0]=(c3x)
        pts[2][1]=(c3y)
        
        pts[3][0]=(c4x)
        pts[3][1]=(c4y)


        #######################
        source = np.array([
                [c1x, c1y],
                [c2x, c2y],
                [c3x, c2y],
                [c4x, c4y]], dtype = "float32")
        
        print("source",source)
        dst = np.array([
                [172, 138],
                [1130, 138],
                [1130, 610],
                [172, 610]], dtype = "float32")


        #dst = np.array([
         #       [194, 60],
          #      [439, 137],
           #     [203, 287],
            #    [46, 86]], dtype = "float32")        
        
        M = cv2.getPerspectiveTransform(source, dst) 

        
        print ('dest is ',dst)

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

playerPos = open("playerpos1.txt")
playerPosdata  = np.genfromtxt(playerPos, delimiter=",")

corner = open('corner2.txt')
cornerdata  = np.genfromtxt(corner, delimiter=",")
ball.close()
corner.close()

print("balldata",balldata)
M = getMatrixOfFrame(0,cornerdata)
#for testing######
frameNo = 380
x = balldata[0][1]
y = balldata[0][2]
#############

        
        
#############

#M =getMatrixOfFrame(0,cornerdata)
#print("M",M)
#for i in range(frameNo):
for i in range(0,14):
        M = getMatrixOfFrame(i,cornerdata)
        M1 = getMatrixOfFrame(i+1,cornerdata)
        ###########





        
        print("frame",i)
        print("M",M)

        currFrame = cornerdata[i][0]
        nextFrame = cornerdata[i+1][0]
        diff = nextFrame - currFrame
        diff = int(diff)
        print("currFrame",currFrame)
        print("nextFrame",nextFrame)
        print("diff",diff)


        new_x = playerPosdata[i][1]
        new_y = playerPosdata[i][2]        
        for j in range(0,diff):
                print ("****************frane:",cornerdata[i][0]+j)
                	# grab the frame from the video stream and resize it to have a
                # maximum width of 300 pixels
                        #frame = vs.read()
                frame = cv2.imread('volley_field.jpg')
                #frame = cv2.imread('data/beachVolleyball1/1.jpg')
                #frame = imutils.resize(frame, width=600)
                # check if the writer is None
                if writer is None:
                        # store the image dimensions, initialzie the video writer,
                        # and construct the zeros array
                        (h, w) = frame.shape[:2]
                        writer = cv2.VideoWriter("example.avi", fourcc, 10,
                                (w , h ), True)
                        zeros = np.zeros((h, w), dtype="uint8")
                color1 = (256, 0, 0)
                color2 = (0, 256, 0)

                x_diff1 = (playerPosdata[i+1][0] - playerPosdata[i][0])/float(diff)
                y_diff1 = (playerPosdata[i+1][1] - playerPosdata[i][1])/float(diff)
                
                x_diff2 = (playerPosdata[i+1][2] - playerPosdata[i][2])/float(diff)
                y_diff2 = (playerPosdata[i+1][3] - playerPosdata[i][3])/float(diff)
                
                x_diff3 = (playerPosdata[i+1][4] - playerPosdata[i][4])/float(diff)
                y_diff3 = (playerPosdata[i+1][5] - playerPosdata[i][5])/float(diff)
                
                x_diff4 = (playerPosdata[i+1][6] - playerPosdata[i][6])/float(diff)
                y_diff4 = (playerPosdata[i+1][7] - playerPosdata[i][7])/float(diff)

                #print("pos nex x",playerPosdata[i+1][0])
                #print("pos curr x",playerPosdata[i][0])
                #print("x_diff1",x_diff1)
                #print("y_diff1",y_diff1)

                currentFrame = cornerdata[i][0]+j
                new_x1 = playerPosdata[i][0]
                new_y1 = playerPosdata[i][1]
                new_x2 = playerPosdata[i][2]
                new_y2 = playerPosdata[i][3]
                new_x3 = playerPosdata[i][4]
                new_y3 = playerPosdata[i][5]
                new_x4 = playerPosdata[i][6]
                new_y4 = playerPosdata[i][7]

                pos1 = (new_x1,new_y1)
                pos2 = (new_x2,new_y2)
                pos3 = (new_x3,new_y3)
                pos4 = (new_x4,new_y4)
                
                new2_x1 = playerPosdata[i+1][0]
                new2_y1 = playerPosdata[i+1][1]
                new2_x2 = playerPosdata[i+1][2]
                new2_y2 = playerPosdata[i+1][3]
                new2_x3 = playerPosdata[i+1][4]
                new2_y3 = playerPosdata[i+1][5]
                new2_x4 = playerPosdata[i+1][6]
                new2_y4 = playerPosdata[i+1][7]

                npos1 = (new2_x1,new2_y1)
                npos2 = (new2_x2,new2_y2)
                npos3 = (new2_x3,new2_y3)
                npos4 = (new2_x4,new2_y4)

                #print("pos1",pos1)
                #print("pos2",pos2)
                #print("pos3",pos3)
                #print("pos4",pos4)
                

                ####
                #original = np.array([(point, point2)], dtype=np.float)
                #converted = cv2.perspectiveTransform(original, M)
                #npoint1 = (int(converted[0][0].item(0)),int(converted[0][0].item(1)))
                #print("converted",converted)
                #npoint2 = (int(converted[0][1].item(0)),int(converted[0][1].item(1)))
                #print("pts at",npoint1)
                ####
                original1 = np.array([(pos1, pos2)], dtype=np.float)
                original2 = np.array([(pos3, pos4)], dtype=np.float)

                noriginal1 = np.array([(npos1, npos2)], dtype=np.float)
                noriginal2 = np.array([(npos3, npos4)], dtype=np.float)
                
                converted1 = cv2.perspectiveTransform(original1, M)
                converted2 = cv2.perspectiveTransform(original2, M)

                nconverted1 = cv2.perspectiveTransform(original1, M1)
                nconverted2 = cv2.perspectiveTransform(original2, M1)
                #print("converted1",converted1)
                n1point1 = (int(converted1[0][0].item(0)),int(converted1[0][0].item(1)))
                n1point2 = (int(converted1[0][1].item(0)),int(converted1[0][1].item(1)))
                n2point1 = (int(nconverted1[0][0].item(0)),int(nconverted1[0][0].item(1)))
                n2point2 = (int(nconverted1[0][1].item(0)),int(nconverted1[0][1].item(1)))
                #print("pts1 at",n1point1)
                #print("pts2 at",n1point2)


                #print("converted2",converted2)
                n1point3 = (int(converted2[0][0].item(0)),int(converted2[0][0].item(1)))
                n1point4 = (int(converted2[0][1].item(0)),int(converted2[0][1].item(1)))
                n2point3 = (int(nconverted2[0][0].item(0)),int(nconverted2[0][0].item(1)))
                n2point4 = (int(nconverted2[0][1].item(0)),int(nconverted2[0][1].item(1)))                
                #print("pts3 at",n1point3)
                #print("pts4 at",n1point4)                
                
                #cv2.circle(frame, npoint1, 5, color, -1,4,1)
                #cv2.circle(frame, npoint2, 5, color, -1,4,1)
                pt1diffx = (n2point1[0]- n1point1[0])/float(diff)
                pt1diffy = (n2point1[1] - n1point1[1])/float(diff)
                pt2diffx = (n2point2[0] - n1point2[0])/float(diff)
                pt2diffy = (n2point2[1] - n1point2[1])/float(diff)
                pt3diffx = (n2point3[0] - n1point3[0])/float(diff)
                pt3diffy = (n2point3[1] - n1point3[1])/float(diff)
                pt4diffx = (n2point4[0] - n1point4[0])/float(diff)
                pt4diffy = (n2point4[1] - n1point4[1])/float(diff)

                n1point1 = (int((n1point1[0])+(pt1diffx*j)),int((n1point1[1])+(pt1diffy*j)))
                n1point2 = (int((n1point2[0])+(pt2diffx*j)),int((n1point2[1])+(pt2diffy*j)))
                n1point3 = (int((n2point2[0])+(pt3diffx*j)),int((n2point3[1])+(pt3diffy*j)))
                n1point4 = (int((n2point4[0])+(pt4diffx*j)),int((n2point4[1])+(pt4diffy*j)))



                
                

                
                cv2.circle(frame, n1point1, 10, color1, -1,4,1)
                cv2.circle(frame, n1point2, 10, color1, -1,4,1)
                cv2.circle(frame, n1point3, 10, color2, -1,4,1)
                cv2.circle(frame, n1point4, 10, color2, -1,4,1)
                writer.write(frame)

                # show the frames
                #cv2.imshow("Frame", frame)
                cv2.imshow("Output", frame)
                key = cv2.waitKey(1) & 0xFF

                # for testing
                #plt.figure()
                #plt.imshow(frame,cmap='gray')
                #plt.hold(True)
                #print("shape1",frame.shape)
                #print("shape2",frame[0].shape)
                #plt.scatter(1,2,color='blue')
                #plt.show()

                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                        break
                if key == ord("p"):
                        time.sleep(10)
                
	

	
	

# do a bit of cleanup
print("[INFO] cleaning up...")
#cv2.destroyAllWindows()
#vs.stop()
writer.release()

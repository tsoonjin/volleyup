#!/usr/bin/env python
from __future__ import print_function
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import matplotlib.pyplot as plt


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
                [c3x, c3y],
                [c4x, c4y]], dtype = "float32")
        
        print("source",source)
        dst = np.array([
                [172, 138],
                [1130, 138],
                [1130, 610],
                [172, 610]], dtype = "float32")

        dst0 = np.array([
                [650, 138],
                [1130, 138],
                [1130, 610],
                [650, 610]], dtype = "float32")
        
        dst1 = np.array([
                [172, 138],
                [650, 138],
                [650, 610],
                [172, 610]], dtype = "float32")        
        
        ################################## for sample 2
        dst = np.array([
                [521, 171],
                [521, 730],
                [204, 735],
                [204,171]], dtype = "float32")

        dst1 = np.array([
                [521, 171],
                [521, 454],
                [204, 454],
                [204,171]], dtype = "float32")

        dst0 = np.array([
                [521, 454],
                [521, 730],
                [204, 735],
                [204,454]], dtype = "float32")

        ################
        
        if half == 2:
                print("half == 2")
                M = cv2.getPerspectiveTransform(source, dst)
                
        if half == 0:
                print("half == 0")                
                M = cv2.getPerspectiveTransform(source, dst0)
                
        if half == 1:
                print("half == 1")                
                M = cv2.getPerspectiveTransform(source, dst1)
                
        #M = cv2.getPerspectiveTransform(source, dst)
        
        print ('dest is ',dst)

        return M





print("starting")
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
writer = None
(h, w) = (None, None)
zeros = None


#read player and ball movements, now just simulate
reader = open('movement.txt')


ball = open("ball_v1.txt")
balldata  = np.genfromtxt(ball, delimiter=",")

playerPos = open("playerpos7.txt")
#playerpos7
playerPosdata  = np.genfromtxt(playerPos, delimiter=",")

corner = open('cornerData7.txt')
cornerdata  = np.genfromtxt(corner, delimiter=",")
ball.close()
corner.close()

print("balldata",balldata)
M = getMatrixOfFrame(0,cornerdata)




       
#############

#M =getMatrixOfFrame(0,cornerdata)
#print("M",M)
#for i in range(frameNo):
for i in range(0,9):
        M = getMatrixOfFrame(i,cornerdata)
        ###########       
        print("line:",i)
        print("M",M)

        currFrame = cornerdata[i][0]
        nextFrame = cornerdata[i+1][0]
        diff = nextFrame - currFrame
        diff = int(diff)
        print("currFrame",currFrame)
        print("nextFrame",nextFrame)
        print("diff",diff)

        
        for j in range(0,(diff/5)-1):
                currentFrame = int(cornerdata[i][0]+(j*5))
                nextSetFrame = int(cornerdata[i][0]+(j*5)+5)
                print ("****************frane:",currentFrame)
                # grab the frame from the video stream and resize it to have a
                # maximum width of 300 pixels
                        #frame = vs.read()
                frame = cv2.imread('sample2.jpg')
                #frame = cv2.imread('data/beachVolleyball1/1.jpg')
                #frame = imutils.resize(frame, width=600)
                # check if the writer is None
                if writer is None:
                        # store the image dimensions, initialzie the video writer,
                        # and construct the zeros array
                        (h, w) = frame.shape[:2]
                        writer = cv2.VideoWriter("video7.avi", fourcc, 50,
                                (w , h ), True)
                        zeros = np.zeros((h, w), dtype="uint8")
                color1 = (256, 0, 0)
                color2 = (0, 256, 0)



                #print("pos nex x",playerPosdata[i+1][0])
                #print("pos curr x",playerPosdata[i][0])
                #print("x_diff1",x_diff1)
                #print("y_diff1",y_diff1)

                new_x1 = playerPosdata[currentFrame][0]
                new_y1 = playerPosdata[currentFrame][1]
                new_x2 = playerPosdata[currentFrame][2]
                new_y2 = playerPosdata[currentFrame][3]
                new_x3 = playerPosdata[currentFrame][4]
                new_y3 = playerPosdata[currentFrame][5]
                new_x4 = playerPosdata[currentFrame][6]
                new_y4 = playerPosdata[currentFrame][7]

                pos1 = (new_x1,new_y1)
                pos2 = (new_x2,new_y2)
                pos3 = (new_x3,new_y3)
                pos4 = (new_x4,new_y4)


                new1_x1 = playerPosdata[nextSetFrame][0]
                new1_y1 = playerPosdata[nextSetFrame][1]
                new1_x2 = playerPosdata[nextSetFrame][2]
                new1_y2 = playerPosdata[nextSetFrame][3]
                new1_x3 = playerPosdata[nextSetFrame][4]
                new1_y3 = playerPosdata[nextSetFrame][5]
                new1_x4 = playerPosdata[nextSetFrame][6]
                new1_y4 = playerPosdata[nextSetFrame][7]

                npos1 = (new_x1,new_y1)
                npos2 = (new_x2,new_y2)
                npos3 = (new_x3,new_y3)
                npos4 = (new_x4,new_y4)
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
                converted1 = cv2.perspectiveTransform(original1, M)
                converted2 = cv2.perspectiveTransform(original2, M)

                noriginal1 = np.array([(npos1, npos2)], dtype=np.float)
                noriginal2 = np.array([(npos3, npos4)], dtype=np.float)
                nconverted1 = cv2.perspectiveTransform(noriginal1, M)
                nconverted2 = cv2.perspectiveTransform(noriginal2, M)
                #print("converted1",converted1)
                
                #print("pts1 at",n1point1)
                #print("pts2 at",n1point2)
                
                #if int(converted1[0][0].item(1)) < 454:
                 #       n1point1 = (int(converted1[0][0].item(0)),int(454))
                #if int(converted1[0][1].item(1)) < 454:
                 #       n1point2 = (int(converted1[0][0].item(0)),int(454))

                #print("converted2",converted2)

                #print("pts3 at",n1point3)
                #print("pts4 at",n1point4)                
                #if int(converted2[0][0].item(1)) > 454:
                 #       n1point3 = (int(converted1[0][0].item(0)),int(454))
                #if int(converted2[0][1].item(1)) > 454:
                 #       n1point4 = (int(converted1[0][0].item(0)),int(454))                
                #cv2.circle(frame, npoint1, 5, color, -1,4,1)
                #cv2.circle(frame, npoint2, 5, color, -1,4,1)


                #final pts in current frame        
                n1point1 = (int(converted1[0][0].item(0)),int(converted1[0][0].item(1)))
                n1point2 = (int(converted1[0][1].item(0)),int(converted1[0][1].item(1)))
                n1point3 = (int(converted2[0][0].item(0)),int(converted2[0][0].item(1)))
                n1point4 = (int(converted2[0][1].item(0)),int(converted2[0][1].item(1)))

                n2point1 = (int(nconverted1[0][0].item(0)),int(nconverted1[0][0].item(1)))
                n2point2 = (int(nconverted1[0][1].item(0)),int(nconverted1[0][1].item(1)))
                n2point3 = (int(nconverted2[0][0].item(0)),int(nconverted2[0][0].item(1)))
                n2point4 = (int(nconverted2[0][1].item(0)),int(nconverted2[0][1].item(1)))  

                pt1diffx = (n2point1[0] - n1point1[0])/float(diff)
                pt1diffy = (n2point1[1] - n1point1[1])/float(diff)

                                
                pt2diffx = (n2point2[0] - n1point2[0])/float(diff)
                pt2diffy = (n2point2[1] - n1point2[1])/float(diff)

                pt3diffx = (n2point3[0] - n1point3[0])/float(diff)
                pt3diffy = (n2point3[1] - n1point3[1])/float(diff)

                pt4diffx = (n2point4[0] - n1point4[0])/float(diff)
                pt4diffy = (n2point4[1] - n1point4[1])/float(diff)

                


                
                n1point1 = (int(converted1[0][0].item(0)),int(converted1[0][0].item(1)))
                n1point2 = (int(converted1[0][1].item(0)),int(converted1[0][1].item(1)))

                n1point3 = (int(converted2[0][0].item(0)),int(converted2[0][0].item(1)))
                n1point4 = (int(converted2[0][1].item(0)),int(converted2[0][1].item(1)))




                for k in range(0,j):
                        result_n1point1 = (int((n1point1[0])+(pt1diffx*k)),int((n1point1[1])+(pt1diffy*k)))
                        result_n1point2 = (int((n1point2[0])+(pt2diffx*k)),int((n1point2[1])+(pt2diffy*k)))
                        result_n1point3 = (int((n1point2[0])+(pt3diffx*k)),int((n1point3[1])+(pt3diffy*k)))
                        result_n1point4 = (int((n1point4[0])+(pt4diffx*k)),int((n1point4[1])+(pt4diffy*k)))

                        
                        cv2.circle(frame, result_n1point1, 10, color1, -1,4,1)
                        cv2.circle(frame, result_n1point2, 10, color1, -1,4,1)
                        cv2.circle(frame, result_n1point3, 10, color2, -1,4,1)
                        cv2.circle(frame, result_n1point4, 10, color2, -1,4,1)
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

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
import numpy.linalg as la

def getMatrixOfFrame(i,cornerdata):
        ######################
        # the way half is represented
        #################
        #       #       #
        #   1   #   0   #
        #       #       #
        #################

        #################
        #               #
        #      2        #
        #               #
        #################        
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
                M = cv2.getPerspectiveTransform(source, dst)
                
        if half == 0:
                M = cv2.getPerspectiveTransform(source, dst0)
                
        if half == 1:
                M = cv2.getPerspectiveTransform(source, dst1)
                
        print ("M",M)
        
        return M




# construct the argument parse and parse the arguments





fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
writer = None
(h, w) = (None, None)
zeros = None



#read player and ball movements, now just simulate
reader = open('movement.txt')


#ball = open("ball_v1.txt")
#balldata  = np.genfromtxt(ball, delimiter=",")

for num in range (1,8):
        print("**********************converting video:    ",num)
        print("**********************converting video:    ",num)
        print("**********************converting video:    ",num)
        print("**********************converting video:    ",num)
        print("**********************converting video:    ",num)
        print("**********************converting video:    ",num)
        print("**********************converting video:    ",num)
        print("**********************converting video:    ",num)        
        num1 = str(num)
        playerPos = open("playerpos"+num1+".txt")
        playerPosdata  = np.genfromtxt(playerPos, delimiter=",")

        corner = open("cornerData"+num1+".txt")
        cornerdata  = np.genfromtxt(corner, delimiter=",")
        #ball.close()
        corner.close()
        #273,300,437,300
        #273,628,437,628

        (row,col) = cornerdata.shape
        for i in range(0,row-1):
                #matrix for current set frame
                M = getMatrixOfFrame(i,cornerdata)
                #matrix for next set frame
                M1 = getMatrixOfFrame(i+1,cornerdata)





                                
        #273,300,437,300
        #273,628,437,628


        #100 100 100 72
        #200 100 200 72

                                
                print("line: ",i+1)
                currFrame = cornerdata[i][0]
                nextFrame = cornerdata[i+1][0]
                diff = nextFrame - currFrame
                diff = int(diff)
                #print("currFrame",currFrame)
                #print("nextFrame",nextFrame)
                #print("diff",diff)

                new_x1 = playerPosdata[i][0]
                new_y1 = playerPosdata[i][1]
                new_x2 = playerPosdata[i][2]
                new_y2 = playerPosdata[i][3]
                new_x3 = playerPosdata[i][4]
                new_y3 = playerPosdata[i][5]
                new_x4 = playerPosdata[i][6]
                new_y4 = playerPosdata[i][7]

                ##construct 4 points in current frame
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

                ##construct 4 points in the next frame
                npos1 = (new2_x1,new2_y1)
                npos2 = (new2_x2,new2_y2)
                npos3 = (new2_x3,new2_y3)
                npos4 = (new2_x4,new2_y4)



                ##construct data for current frame
                original1 = np.array([(pos1, pos2)], dtype=np.float)
                original2 = np.array([(pos3, pos4)], dtype=np.float)

                ##construct data for next frame
                noriginal1 = np.array([(npos1, npos2)], dtype=np.float)
                noriginal2 = np.array([(npos3, npos4)], dtype=np.float)
                
                #print ("original1",original1)
                #print ("original2",original2)
                #print ("noriginal1",noriginal1)
                #print ("noriginal2",noriginal2)

                #convert result for p1 p2 in current frame        
                converted1 = cv2.perspectiveTransform(original1, M)
                
                #convert result for p3 p4 in current frame
                converted2 = cv2.perspectiveTransform(original2, M)

                #convert result for p1 p2 in next frame         
                nconverted1 = cv2.perspectiveTransform(noriginal1, M1)
                
                #convert result for p3 p4 in next frame  
                nconverted2 = cv2.perspectiveTransform(noriginal2, M1)

                #print("converted1",converted1)
                #print("nconverted1",nconverted1)

                #final pts in current frame        
                n1point1 = (int(converted1[0][0].item(0)),int(converted1[0][0].item(1)))
                n1point2 = (int(converted1[0][1].item(0)),int(converted1[0][1].item(1)))
                n1point3 = (int(converted2[0][0].item(0)),int(converted2[0][0].item(1)))
                n1point4 = (int(converted2[0][1].item(0)),int(converted2[0][1].item(1)))



                #print("converted2",converted2)
                #print("nconverted2",nconverted2)

                #fina pts in next frame        
                n2point1 = (int(nconverted1[0][0].item(0)),int(nconverted1[0][0].item(1)))
                n2point2 = (int(nconverted1[0][1].item(0)),int(nconverted1[0][1].item(1)))
                n2point3 = (int(nconverted2[0][0].item(0)),int(nconverted2[0][0].item(1)))
                n2point4 = (int(nconverted2[0][1].item(0)),int(nconverted2[0][1].item(1)))





                #for sample 2


                
                if playerPosdata[i][0] == -999999:
                        n1point1 = (237,300)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
                if playerPosdata[i][2] == -999999:
                        n1point2 = (437,300)                        
                if playerPosdata[i][4] == -999999:
                        n1point3 = (237,628)
                if playerPosdata[i][6] == -999999:
                        n1point4 = (437,628)

                if playerPosdata[i+1][0] == -999999:
                        n2point1 = (237,300)
                if playerPosdata[i+1][2] == -999999:
                        n2point2 = (437,300)                        
                if playerPosdata[i+1][4] == -999999:
                        n2point3 = (237,628)
                if playerPosdata[i+1][6] == -999999:
                        n2point4 = (437,628)

                #print("pts1 at",n1point1)
                #print("pts2 at",n1point2)
                #print("pts3 at",n1point3)
                #print("pts4 at",n1point4)
                #print("new next set frame pts1 at",n2point1)
                #print("new next set frame pts2 at",n2point2) 
                #print("new next set frame pts3 at",n2point3)
                #print("new next set frame pts4 at",n2point4)                        

                pt1diffx = (n2point1[0] - n1point1[0])/float(diff)
                pt1diffy = (n2point1[1] - n1point1[1])/float(diff)

                                
                pt2diffx = (n2point2[0] - n1point2[0])/float(diff)
                pt2diffy = (n2point2[1] - n1point2[1])/float(diff)

                pt3diffx = (n2point3[0] - n1point3[0])/float(diff)
                pt3diffy = (n2point3[1] - n1point3[1])/float(diff)

                pt4diffx = (n2point4[0] - n1point4[0])/float(diff)
                pt4diffy = (n2point4[1] - n1point4[1])/float(diff)

                

                print("diff is",diff)        
                for j in range(0,diff):
                        #print ("****************frame:",cornerdata[i][0]+j)
                        frame = cv2.imread('volley_field.jpg')
                        frame = cv2.imread('sample2.jpg')
                        #frame = cv2.imread('data/beachVolleyball1/1.jpg')
                        #frame = imutils.resize(frame, width=600)
                        # check if the writer is None
                        if writer is None:
                                # store the image dimensions, initialzie the video writer,
                                # and construct the zeros array
                                (h, w) = frame.shape[:2]

                                print("num",num1)
                                writer = cv2.VideoWriter("final_video"+num1+".avi", fourcc, 60,
                                        (w , h ), True)
                                zeros = np.zeros((h, w), dtype="uint8")
                        color1 = (256, 0, 0)
                        color2 = (0, 256, 0)

                        currentFrame = cornerdata[i][0]+j

                                      
                        
                        #cv2.circle(frame, npoint1, 5, color, -1,4,1)
                        #cv2.circle(frame, npoint2, 5, color, -1,4,1)


                        result_n1point1 = (int((n1point1[0])+(pt1diffx*j)),int((n1point1[1])+(pt1diffy*j)))
                        result_n1point2 = (int((n1point2[0])+(pt2diffx*j)),int((n1point2[1])+(pt2diffy*j)))
                        result_n1point3 = (int((n1point2[0])+(pt3diffx*j)),int((n1point3[1])+(pt3diffy*j)))
                        result_n1point4 = (int((n1point4[0])+(pt4diffx*j)),int((n1point4[1])+(pt4diffy*j)))
                        #print("final pts1 at",n1point1)
                        #print("final pts2 at",n1point2)
                        #print("final pts3 at",n1point3)
                        #print("final pts4 at",n1point4) 
                        
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
        writer = None
	

	
	

# do a bit of cleanup
print("[INFO] cleaning up...")
#cv2.destroyAllWindows()
#vs.stop()
writer.release()

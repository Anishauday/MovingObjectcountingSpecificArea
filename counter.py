import datetime
import math
import cv2
import numpy as np
import imutils
import sys

#global variables
width = 0
height = 0
entranceCounter = 0
exitCounter = 0
minContourArea = 2500  #Adjust this value according to your usage
binarizationThreshold = 38  #Adjust this value according to your usage
offsetRefLines = 200  # From center of frame
pixelTolerance = 10 # Tolerance zone width(/2) in pixels
coordXEntranceLine = 0
coordXExitLine = 0

#Check if an object in entering in monitored zone
def CheckEntranceLineCrossing(x, coordXEntranceLine, coordXExitLine, height):
    absDistance = abs(x - coordXEntranceLine)    
    print(x)

    if ((absDistance <= pixelTolerance) and (x < coordXExitLine) and (x < (height-200))):
        return 1
    else:
        return 0

#Check if an object is exiting from monitored zone
def CheckExitLineCrossing(x, coordXEntranceLine, coordXExitLine, height):
    absDistance = abs(x - coordXExitLine)    

    if ((absDistance <= pixelTolerance) and (x > coordXEntranceLine) and (x < height)):
        return 1
    else:
        return 0

#camera = cv2.VideoCapture("C:/Users/662748/Downloads/Object tracking dataset/cartrim.mp4")
camera = cv2.VideoCapture(0)

#force 640x480 webcam resolution
camera.set(3,640)
camera.set(4,480)

referenceFrame = None

#The webcam maybe get some time / captured frames to adapt to ambience lighting. 
#For this reason, some frames are grabbed and discarded.
for i in range(0,10):
    (grabbed, frame) = camera.read()
    height = np.size(frame,0)
    width = np.size(frame,1)
    
    
    #refline positions (rounded to get int value)
    coordXEntranceLine = round((width / 2) - offsetRefLines)
    #print(coordXEntranceLine)
    #print(width)
    #print(offsetRefLines)
    coordXExitLine = round((width / 2) + offsetRefLines)
    #print(coordXExitLine)
    #print(width)
    #print(offsetRefLines)
    #print(coordXEntranceLine)
    #print(coordXExitLine)

while True:    
    (grabbed, frame) = camera.read()

    #if cannot grab a frame, this program ends here.
    if not grabbed:
        break

    #gray-scale convertion and Gaussian blur filter applying
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayFrame = cv2.GaussianBlur(grayFrame, (21, 21), 0)
    
    if referenceFrame is None:
        referenceFrame = grayFrame
        continue

    #Background subtraction and image binarization
    frameDelta = cv2.absdiff(referenceFrame, grayFrame)
    frameThresh = cv2.threshold(frameDelta, binarizationThreshold, 255, cv2.THRESH_BINARY)[1]
    
    #Dilate image and find all the contours
    frameThresh = cv2.dilate(frameThresh, None, iterations=2)
    cnts = cv2.findContours(frameThresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    qtyOfContours = 0
    
    disp_frame = frameThresh

    #plot reference lines (entrance and exit lines) 
    cv2.rectangle(disp_frame, (coordXEntranceLine-100,400), (coordXEntranceLine+100,height-200), (255, 0, 0), 2)
    cv2.rectangle(disp_frame, (coordXExitLine-100,400), (coordXExitLine+100,height-200), (255, 0, 0), 2)

    #check all found contours
    for c in cnts:
        #if a contour has small area, it'll be ignored
        if cv2.contourArea(c) < minContourArea:
            continue

        qtyOfContours += 1 

        #draw a rectangle "around" the object
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(disp_frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

        #find object's centroid (rounded to get int value)
        coordXCentroid = round(x+w/2)
        coordYCentroid = round(y+h/2)
        cv2.circle(disp_frame, (coordXCentroid, coordYCentroid), 1, (255, 255, 255), 5)
        
        if (CheckEntranceLineCrossing(coordXCentroid,coordXEntranceLine,coordXExitLine,height)):
            entranceCounter += 1

        if (CheckExitLineCrossing(coordXCentroid,coordXEntranceLine,coordXExitLine,height)):  
            exitCounter += 1
            
    sys.stdout.write("\r")
    sys.stdout.write("Total contours found: " + str(qtyOfContours))
    sys.stdout.flush()

    #Write entrance and exit counter values on frame and show it
    cv2.putText(disp_frame, "Box A: {}".format(str(entranceCounter)), (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 0, 0), 2)
    cv2.putText(disp_frame, "Box B: {}".format(str(exitCounter)), (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 0, 1), 2)
    
    cv2.imshow("Background substracted image", disp_frame)
    cv2.imshow("Original", frame)
    cv2.waitKey(1);


# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()

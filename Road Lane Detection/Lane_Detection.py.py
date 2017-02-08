import cv2
import numpy as np

img = cv2.imread('R1.jpg',1)
#blur the image to remove background noise
blur=cv2.blur(img,(5,5))

#Convert to gray scale and select region of interest
gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
roi = gray[img.shape[0]/2:img.shape[0]/2+img.shape[0]/3, 50:img.shape[1]]

#Detect edges and find the lanes
edges = cv2.Canny(roi,50,150,apertureSize = 3)
cv2.imshow('canny',edges)

minLineLength = 50
maxLineGap = 1
lines = cv2.HoughLinesP(edges,1,np.pi/180,30,minLineLength,maxLineGap)
if lines!=None:
    for x in range(24):
        for x1,y1,x2,y2 in lines[x]:
            cv2.line(img,(x1+50,y1+img.shape[0]/2),(x2+50,y2+img.shape[0]/2),(0,255,0),3)

cv2.imshow('lane detection.jpg',img)
cv2.waitKey(0)
cv2.destroyAllWindows()




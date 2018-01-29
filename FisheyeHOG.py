# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 21:00:46 2017

@author: otalab
"""

import cv2
import time
import numpy as np

hog = cv2.FisheyeHOGDescriptor()
hog.setSVMDetector(cv2.FisheyeHOGDescriptor_getDefaultPeopleDetector())
hog.setAngleMatrix((768,768))
hogorig = cv2.HOGDescriptor()
hogorig.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
#testimg = cv2.imread('image/multiscale.jpg')
testimg = cv2.imread('image/pedestrian_60step.jpg')
template = cv2.imread('image/pedestriantight.jpg')
draw = testimg.copy()

#grad_t, qangles_t = hogorig.computeGradient(template)
#grad, qangles = hog.computeGradient(testimg)
#cv2.imshow('template', qangles_t[:,:,0]*16-1)
#cv2.imshow('bins',qangles[:,:,0]*16-1)
#cv2.waitKey()

t0=time.time();
(rects2, weights2, des_detect) = hog.detect(testimg, padding=(0,0));
t1=time.time()
print t1-t0

#t0=time.time();
#(rects3, weights3) = hog.detectMultiScale(testimg, padding=(0,0), scale=1.05, winStride=(4,4));
#t1=time.time()

for rect in rects2:
    box = cv2.cv.BoxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(draw,[box],0,(0,0,255),2)
    
#for rect in rects3:
#    box = cv2.cv.BoxPoints(rect)
#    box = np.int0(box)
#    cv2.drawContours(draw,[box],0,(0,255,0),2)
cv2.imshow("fisheye",draw)
#print t1-t0
cv2.imwrite('output/detected.jpg',draw)
cv2.waitKey(0)

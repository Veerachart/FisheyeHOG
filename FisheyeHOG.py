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
testimg = cv2.imread('image/pedestriansimulated.jpg')
draw = testimg.copy()

t0=time.time();
(rects2, weights2) = hog.detect(testimg, padding=(0,0));
t1=time.time()

for rect in rects2:
    box = cv2.cv.BoxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(draw,[box],0,(0,0,255),2)
cv2.imshow("fisheye",draw)
print t1-t0
cv2.waitKey(0)
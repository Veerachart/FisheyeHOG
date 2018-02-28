# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 11:51:54 2018

@author: otalab
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

hog = cv2.FisheyeHOGDescriptor()
hog.setSVMDetector(cv2.FisheyeHOGDescriptor_getDefaultPeopleDetector())
hog.setAngleMatrix((768,768))
hogorig = cv2.HOGDescriptor()
hogorig.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
#testimg = cv2.imread('image/multiscale.jpg')
testimg = cv2.imread('image/pedestrian_single.jpg')
tight_img = cv2.imread('image/pedestriantight.jpg')
hogorig = cv2.HOGDescriptor()
hogorig.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
#(points,weights,des_detect) = hogorig.detect(tight_img,winStride=(4,4),padding=(0,0))
#des_detect=des_detect.flatten()

des_orig = hogorig.compute(tight_img, padding=(0,0))

rot_angles = np.arange(0,360,4)
rot_angles = [340]

rows, cols, channels = testimg.shape

for rot_angle in rot_angles:
    print(rot_angle)
    
    M = cv2.getRotationMatrix2D((cols/2, rows/2), rot_angle, 1)
    dst = cv2.warpAffine(testimg,M,(cols,rows))
    
    draw = dst.copy()
    
    (rects, weights, des_detect) = hog.detect(dst, padding=(0,0));
    des_detect = des_detect.flatten()
    
    for rect in rects:
        box = cv2.cv.BoxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(draw,[box],0,(0,0,255),2)
        
    cv2.imshow("Fisheye", draw)
    
#    num_detect = len(rects)
#    des_compute = hog.compute(dst, rects)
#    des_compute = des_compute.flatten()
#    for i in range(num_detect):
#        fig = plt.figure();
#        plt.plot(des_compute[3780*i:3780*(i+1)]-des_detect[3780*i:3780*(i+1)])
#        fig.show()
        
    cv2.waitKey(0)
    plt.close('all')
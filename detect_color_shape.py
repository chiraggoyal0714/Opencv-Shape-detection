
from pyimagesearch.shapedetector import ShapeDetector
from pyimagesearch.colorlabeler import ColorLabeler
import argparse
import imutils
import cv2
import numpy as np


image = cv2.imread("shapesPhoto.png")
l,h,c=image.shape
#print l,h,c
r = 150.0 / image.shape[1]
dim = (150, int(image.shape[0] * r))
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
lower_white = np.array([220, 220, 220], dtype=np.uint8)
upper_white = np.array([255, 255, 255], dtype=np.uint8)
mask = cv2.inRange(resized, lower_white, upper_white) 
res = cv2.bitwise_not(resized, resized, mask)
#cv2.imshow('res', res)
resized = imutils.resize(res, width=898)
ratio = image.shape[0] / float(resized.shape[0])

blurred = cv2.GaussianBlur(resized, (5, 5), 0)
gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
#cv2.imshow("Thresh", thresh)

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]


sd = ShapeDetector()
cl = ColorLabeler()

for c in cnts:
	# compute the center of the contour
	M = cv2.moments(c)
	cX = int((M["m10"] / M["m00"]) * ratio)
	cY = int((M["m01"] / M["m00"]) * ratio)

	# detect the shape of the contour and label the color
	shape = sd.detect(c)
	color = cl.label(lab, c)

	c = c.astype("float")
	c *= ratio
	c = c.astype("int")
	text = "{} {} {} {}".format(color, shape, cX,cY)
	cv2.drawContours(image, [c], -1, (255, 255, 255), 2)
	cv2.putText(image, text, (cX, cY),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

	cv2.imshow("Image", image)
	cv2.waitKey(0)

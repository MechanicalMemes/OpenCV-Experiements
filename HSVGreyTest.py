import cv2
import numpy

raw = cv2.imread("images/test4.jpg")

hsv = cv2.cvtColor(raw,cv2.COLOR_BGR2HSV)



perfer = numpy.array([120,200,100]) #Lighter Blue Range


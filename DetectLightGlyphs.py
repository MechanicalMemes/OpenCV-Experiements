import cv2
import numpy

raw = cv2.imread("images/test2.jpg")

hsv = cv2.cvtColor(raw,cv2.COLOR_BGR2HSV)

lower = numpy.array([15 , 0, 133])
upper = numpy.array([190, 30 ,159])

mask = cv2.inRange(hsv, lower, upper)
kernel = numpy.ones((3,3), numpy.uint8)
mask = cv2.erode(mask, kernel)
mask = cv2.dilate(mask, kernel)


#mask = cv2.bilateralFilter(mask, 11, 17, 17)
#mask = cv2.blur(mask, (10,10))
im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)




for c in contours:
    cnt = c
    if cv2.contourArea(c) >= 500:
        x, y, w, h = cv2.boundingRect(cnt)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 :
            if y > 10:

                cv2.putText(raw, "Cube" + str(x), (x + 20, y + int(h / 2)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
                cv2.rectangle(raw, (x, y), (x + w, y + h), (255, 0, 0), 2)

areas = [cv2.contourArea(c) for c in contours]
max_index = numpy.argmax(areas)
cnt=contours[max_index]

x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(raw,(x,y),(x+w,y+h),(0,255,0),2)
cv2.putText(raw, "Chosen" , (x + 20,y + int(h/2) - 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,255,0))
cv2.imshow("Raw", raw)
cv2.imshow("Mask", mask)
while(True):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
import cv2
import numpy

raw = cv2.imread("images/test1.jpg")

height, width = raw.shape[:2]
grey = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
grey = cv2.GaussianBlur(grey, (3, 3), 1)
grey = cv2.bilateralFilter(grey, 11, 17, 17)
edges = cv2.Canny(grey,20,50)
structure = cv2.getStructuringElement(cv2.MORPH_RECT, (45,45))
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, structure)

img1, cnts, hq = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None

cubeMask = numpy.zeros((height,width), numpy.uint8)

for c in cnts:
    # approximate the contour


    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # if our approximated contour has four points, then
    # we can assume that we have found our screen
    hullPoints = cv2.convexHull(c)
    cv2.polylines(raw, hullPoints,True, (0,255,255), 4)

    if len(approx) <= 5:
       cv2.drawContours(raw, [approx], -1, (255,255,255))


cubeLayer = cv2.bitwise_and(raw, raw, mask=cubeMask)

areas = [cv2.contourArea(c) for c in cnts]
max_index = numpy.argmax(areas)
c=cnts[max_index]

x,y,w,h = cv2.boundingRect(c)

result = int(x + (w/2))

cv2.rectangle(raw,(x,y),(x+w,y+h),(0,255,0),2)
cv2.putText(raw, "Chosen" , (x + 10,y + int(h/2) - 40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,255,0), 2)

cv2.putText(raw, "RESULT: " + str(result), (result,30),0,1,(0,255,0), 1)
cv2.line(raw,(result,0), (result, 1000), (0,255,0), 3)

cv2.imshow("Raw", raw)





cv2.imshow("Cube Layer", cubeLayer)

cv2.imshow("Raw", raw)

cv2.imshow("Edges", edges)

while(True):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
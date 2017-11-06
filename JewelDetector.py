import cv2
import numpy

raw = cv2.imread("images/jewels/001.jpg")
raw = cv2.resize(raw,(800,600))

def DetectBlue(blueIn):
    hsv = cv2.cvtColor(blueIn, cv2.COLOR_BGR2HSV)

    lower = numpy.array([80, 55, 40])
    upper = numpy.array([115, 255, 230])

    mask = cv2.inRange(hsv, lower, upper)
    kernel = numpy.ones((5, 5), numpy.uint8)
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)

    # mask = cv2.bilateralFilter(mask, 11, 17, 17)
    # mask = cv2.blur(mask, (10,10))
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    foundObjects = []
    for c in contours:
        cnt = c
        if cv2.contourArea(c) >= 500:
            x, y, w, h = cv2.boundingRect(cnt)
            if 1 - numpy.abs(w / h) < 0.1:
                cv2.putText(blueIn, "Cube" + str(x), (x + 20, y + int(h / 2)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
                cv2.rectangle(blueIn, (x, y), (x + w, y + h), (255, 255, 255), 2)
                foundObjects.append((x, y, w, h))

    currentMax = -1
    chosenObject = (-1, -1, -1, -1)
    for obj in foundObjects:
        area = obj[0] * obj[1]
        if area > currentMax:
            currentMax = area
            chosenObject = obj

    cv2.rectangle(blueIn, (chosenObject[0], chosenObject[1]),
                  (chosenObject[0] + chosenObject[2], chosenObject[1] + chosenObject[3]), (255, 0, 0), 2)

    cv2.imshow("Raw Blue", blueIn)
    cv2.imshow("Mask Blue", mask)
    return chosenObject[0]

def DetectRed(redIn):
    hsv = cv2.cvtColor(redIn, cv2.COLOR_BGR2HSV)




    lowerMask = cv2.inRange(hsv, numpy.array([0, 150, 100]), numpy.array([10, 255, 255]))
    upperMask = cv2.inRange(hsv, numpy.array([160, 100, 100]), numpy.array([179, 255, 255]))
    # mask = cv2.bilateralFilter(mask, 11, 17, 17)
    # mask = cv2.blur(mask, (10,10))

    mask = cv2.addWeighted(lowerMask,1.0,upperMask,1.0,0.0)
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    foundObjects = []
    for c in contours:
        cnt = c
        if cv2.contourArea(c) >= 500:
            x, y, w, h = cv2.boundingRect(cnt)
            if 1 - numpy.abs(w / h) < 0.1:
                cv2.putText(redIn, "Cube" + str(x), (x + 20, y + int(h / 2)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
                cv2.rectangle(redIn, (x, y), (x + w, y + h), (255, 255, 255), 2)
                foundObjects.append((x, y, w, h))

    currentMax = -1
    chosenRedObject = (-1, -1, -1, -1)
    for obj in foundObjects:
        area = obj[0] * obj[1]
        if area > currentMax:
            currentMax = area
            print("New Max Red Area: " + str(area))
            chosenRedObject = obj

    cv2.rectangle(redIn, (chosenRedObject[0], chosenRedObject[1]),
                  (chosenRedObject[0] + chosenRedObject[2], chosenRedObject[1] + chosenRedObject[3]), (0, 0, 255), 2)

    cv2.imshow("Raw Red", redIn)
    cv2.imshow("mask Red", mask)
    return chosenRedObject[0]

blue = DetectBlue(raw)
red = DetectRed(raw)
print("Red" + str(red))
print("Blue" + str(blue))
final = raw

if blue < red:
    cv2.putText(final, "Blue - Red", (20,50), 0, 1.0,(0,255,0), 2)
if red < blue :
    cv2.putText(final, "Red - Blue", (20,50), 0, 1.0,(0,255,0), 2)
cv2.imshow("Final", final)

while(True):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
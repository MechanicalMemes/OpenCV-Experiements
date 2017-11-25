<<<<<<< HEAD:OLD_DetectRows.py

import numpy
import cv2
#Load Test Image
raw = cv2.imread("images/row1.jpg")

height, width = raw.shape[:2]
#Convert to HSV for color filtering
hsv = cv2.cvtColor(raw,cv2.COLOR_BGR2HSV)

#Prepare a kernal ERODE/DIALTE
kernel = numpy.ones((4,4), numpy.uint8)

#Erode, Dilate, and blur to clean the image up
hsv = cv2.erode(hsv, kernel)
hsv = cv2.dilate(hsv, kernel)
hsv = cv2.blur(hsv, (6,6))


#Set the HSV color rang to filter out
lower = numpy.array([90, 105, 25]) #Darker Blue Range
upper = numpy.array([130,250,150]) #Lighter Blue Range

#mask out all colors that fall outside the range above
mask = cv2.inRange(hsv, lower, upper)

#this is to join together the smaller blue segemnts of the
#crypto box that are split up by the white tape.
#In this case we are allowing the joing along the Y axis to be high.s

structure = cv2.getStructuringElement(cv2.MORPH_RECT, (10,200))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, structure)

im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


boxes = [] #We use this to store our boxes

#Filter and generate bounding boxes that will later become our pillars.
for c in contours:
    cnt = c

    if cv2.contourArea(cnt) > 1000: #Filter by area
        rect = cv2.minAreaRect(cnt)
        ((x, y), (w, h), angle) = rect

        ratio = max(numpy.abs(h / w), numpy.abs(w/h))

        box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
        box = numpy.int0(box)
        cv2.putText(raw, "Height: " + str(round(h, 3)), (int(x) - 10, int(y) - 15), 0, 0.4, (0, 255, 255), 1)
        cv2.putText(raw, "Width: " + str(round(w, 3)), (int(x) - 10, int(y) - 30), 0, 0.4, (0, 255, 255), 1)
        cv2.putText(raw, "Ratio: " + str(round(ratio, 3)), (int(x) - 10, int(y) + 20), 0, 0.4, (0, 255, 255), 1)
        cv2.putText(raw, "Angel: " + str(round(angle, 3)), (int(x) - 10, int(y) + 35), 0, 0.4, (0, 255, 255), 1)
        cv2.putText(raw, "Area: " + str(round(cv2.contourArea(cnt), 3)), (int(x) - 10, int(y) + 50), 0, 0.4,(0, 255, 255), 1)

        if ratio > 3: #Check to see if the box is tall
            boxes.append((x, y, w, h)) #If all true add the box to array
            cv2.putText(raw, "Accepted!", (int(x) - 10, int(y) + -50), 0, 0.5, (0, 255, 0), 1)
            cv2.circle(raw, (int(x), int(y)), 5, (0, 255, 0), 3)
            cv2.drawContours(raw, [box], 0, (0, 255, 0), 2)
        else:
            cv2.putText(raw, "Rejected!" , (int(x) - 10, int(y) + -50), 0, 0.5,(0, 0, 255), 1)
            cv2.circle(raw, (int(x), int(y)), 5, (0, 0, 255), 3)
            cv2.drawContours(raw, [box], 0, (0, 0, 255), 2)



#Sort the boxes by X-Axis coord. This allows us to address each pillar by an index.
def getKey(item):
    return item[0]

boxes = sorted(boxes, key=getKey)


#Slot calulcation logic
def drawSlot(slot):
    leftRow = boxes[slot] #Get the pillar to the left

    rightRow = boxes[slot + 1] #Get the pillar to the right

    leftX = leftRow[0] #Get the X Coord
    rightX = rightRow[0] #Get the X Coord




    drawX = int((rightX - leftX) / 2) + leftX #Calculate the point between the two
    drawY = leftRow[1]  +(leftRow[3] /2)  #Calculate Y Coord. We wont use this in our bot's opetation, buts its nice for drawing

    return (int(drawX),int(drawY))

#Draw Slots
print(len(boxes))
if len(boxes ) >=3:
    left = drawSlot(0)
    center = drawSlot(1)
    right = drawSlot(2)


    cv2.putText(raw, "Left", (left[0] - 10, left[1] - 20), 0,0.8, (0,255,255),2)
    cv2.circle(raw,left, 5,(0,255,255), 3);

    cv2.putText(raw, "Center", (center[0] - 10, center[1] - 20), 0,0.8, (0,255,255),2)
    cv2.circle(raw,center, 5,(0,255,255), 3);

    cv2.putText(raw, "Right", (right[0] - 10, right[1] - 20), 0,0.8, (0,255,255),2)
    cv2.circle(raw,right, 5,(0,255,255), 3);

    bottomCenter = (int(width/2),int(height))
    cv2.line(raw, bottomCenter,left, (255,0,0), 4)

    cv2.line(raw, bottomCenter, (bottomCenter[0], left[1]), (255, 255, 255), 3)

    distanceBetween = bottomCenter[0] - left[0]

    cv2.putText(raw, str(distanceBetween),bottomCenter,0,1, (255,255,255))


cv2.imshow("Raw", raw)
cv2.imshow("Mask", mask)

while(True):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
=======
import numpy
import cv2

# Load Test Image
raw = cv2.imread("images/row3.jpg")

height, width = raw.shape[:2]
# Convert to HSV for color filtering
hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)

# Prepare a kernal ERODE/DIALTE
kernel = numpy.ones((4, 4), numpy.uint8)

# Erode, Dilate, and blur to clean the image up
hsv = cv2.erode(hsv, kernel)
hsv = cv2.dilate(hsv, kernel)
hsv = cv2.blur(hsv, (6, 6))

# Set the HSV color rang to filter out
lower = numpy.array([90, 105, 25])  # Darker Blue Range
upper = numpy.array([130, 250, 150])  # Lighter Blue Range

# mask out all colors that fall outside the range above
mask = cv2.inRange(hsv, lower, upper)

# this is to join together the smaller blue segemnts of the
# crypto box that are split up by the white tape.
# In this case we are allowing the joing along the Y axis to be high.s

structure = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 200))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, structure)

im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

boxes = []  # We use this to store our boxes

# Filter and generate bounding boxes that will later become our pillars.
for c in contours:
    cnt = c

    if cv2.contourArea(cnt) > 1000:  # Filter by area
        rect = cv2.minAreaRect(cnt)
        ((x, y), (w, h), angle) = rect

        ratio = max(numpy.abs(h / w), numpy.abs(w / h))

        box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
        box = numpy.int0(box)
        cv2.putText(raw, "Height: " + str(round(h, 3)), (int(x) - 10, int(y) - 15), 0, 0.4, (0, 255, 255), 1)
        cv2.putText(raw, "Width: " + str(round(w, 3)), (int(x) - 10, int(y) - 30), 0, 0.4, (0, 255, 255), 1)
        cv2.putText(raw, "Ratio: " + str(round(ratio, 3)), (int(x) - 10, int(y) + 20), 0, 0.4, (0, 255, 255), 1)
        cv2.putText(raw, "Angel: " + str(round(angle, 3)), (int(x) - 10, int(y) + 35), 0, 0.4, (0, 255, 255), 1)
        cv2.putText(raw, "Area: " + str(round(cv2.contourArea(cnt), 3)), (int(x) - 10, int(y) + 50), 0, 0.4,
                    (0, 255, 255), 1)

        if ratio > 3:  # Check to see if the box is tall
            boxes.append((x, y, w, h))  # If all true add the box to array
            cv2.putText(raw, "Accepted!", (int(x) - 10, int(y) + -50), 0, 0.5, (0, 255, 0), 1)
            cv2.circle(raw, (int(x), int(y)), 5, (0, 255, 0), 3)
            cv2.drawContours(raw, [box], 0, (0, 255, 0), 2)
        else:
            cv2.putText(raw, "Rejected!", (int(x) - 10, int(y) + -50), 0, 0.5, (0, 0, 255), 1)
            cv2.circle(raw, (int(x), int(y)), 5, (0, 0, 255), 3)
            cv2.drawContours(raw, [box], 0, (0, 0, 255), 2)


# Sort the boxes by X-Axis coord. This allows us to address each pillar by an index.
def getKey(item):
    return item[0]


boxes = sorted(boxes, key=getKey)


# Slot calulcation logic
def drawSlot(slot):
    leftRow = boxes[slot]  # Get the pillar to the left

    rightRow = boxes[slot + 1]  # Get the pillar to the right

    leftX = leftRow[0]  # Get the X Coord
    rightX = rightRow[0]  # Get the X Coord

    drawX = int((rightX - leftX) / 2) + leftX  # Calculate the point between the two
    drawY = leftRow[1] + (
    leftRow[3] / 2)  # Calculate Y Coord. We wont use this in our bot's opetation, buts its nice for drawing

    return (int(drawX), int(drawY))


# Draw Slots
print(len(boxes))
if len(boxes) >= 3:
    left = drawSlot(0)
    center = drawSlot(1)
    right = drawSlot(2)

    cv2.putText(raw, "Left", (left[0] - 10, left[1] - 20), 0, 0.8, (0, 255, 255), 2)
    cv2.circle(raw, left, 5, (0, 255, 255), 3);

    cv2.putText(raw, "Center", (center[0] - 10, center[1] - 20), 0, 0.8, (0, 255, 255), 2)
    cv2.circle(raw, center, 5, (0, 255, 255), 3);

    cv2.putText(raw, "Right", (right[0] - 10, right[1] - 20), 0, 0.8, (0, 255, 255), 2)
    cv2.circle(raw, right, 5, (0, 255, 255), 3);

    bottomCenter = (int(width / 2), int(height))
    cv2.line(raw, bottomCenter, left, (255, 0, 0), 4)

    cv2.line(raw, bottomCenter, (bottomCenter[0], left[1]), (255, 255, 255), 3)

    distanceBetween = bottomCenter[0] - left[0]

    cv2.putText(raw, str(distanceBetween), bottomCenter, 0, 1, (255, 255, 255))

cv2.imshow("Raw", raw)
cv2.imshow("Mask", mask)

while (True):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
>>>>>>> master:DetectRows.py

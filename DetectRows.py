import cv2
import numpy

#Load Test Image
raw = cv2.imread("row2.jpg")

#Convert to HSV for color filtering
hsv = cv2.cvtColor(raw,cv2.COLOR_BGR2HSV)
#Prepare a kernal ERODE/DIALTE
kernel = numpy.ones((5,5), numpy.uint8)

#Erode, Dilate, and blur to clean the image up
hsv = cv2.erode(hsv, kernel)
hsv = cv2.dilate(hsv, kernel)
hsv = cv2.blur(hsv, (6,6))

#Set the HSV color rang to filter out
lower = numpy.array([90, 135, 25]) #Darker Blue Range
upper = numpy.array([130,250,150]) #Lighter Blue Range

#mask out all colors that fall outside the range above
mask = cv2.inRange(hsv, lower, upper)

#Combine sections of the image that are close
#this is to join together the smaller blue segemnts of the
#crypto box that are split up by the white tape.
#In this case we are allowing the joing along the Y axis to be high.s

structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40,100))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, structure)

#Find Contours
im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

boxes = [] #We use this to store our boxes

#Filter and generate bounding boxes that will later become our pillars.
for c in contours:
    cnt = c
    if cv2.contourArea(c) >= 100: #Filter by area
        x, y, w, h = cv2.boundingRect(cnt)

        ratio = numpy.abs(h / w)
        if ratio > 1.5: #Check to see if the box is tall
            boxes.append((x, y, w, h)) #If all true add the box to array


#Sort the boxes by X-Axis coord. This allows us to address each pillar by an index.
def getKey(item):
    return item[0]

boxes = sorted(boxes, key=getKey)

#Draw Rectanlges around boxes
for box in boxes:
    x, y, w, h = box
    cv2.rectangle(raw, (x, y), (x + w, y + h), (255, 0, 0), 2)

#Slot calulcation logic
def drawSlot(slot):
    leftRow = boxes[slot] #Get the pillar to the left
    rightRow = boxes[slot + 1] #Get the pillar to the right

    leftX = leftRow[0] #Get the X Coord
    rightX = rightRow[0] #Get the X Coord

    drawX = int((rightX - leftX) / 2) + leftX #Calculate the point between the two
    drawY = leftRow[3] + leftRow[1] #Calculate Y Coord. We wont use this in our bot's opetation, buts its nice for drawing

    return (drawX, drawY)

#Draw Slots
left = drawSlot(0)
center = drawSlot(1)
right =drawSlot(2)

cv2.putText(raw, "Left", (left[0] - 10, left[1] - 20), 0,0.8, (0,255,255),2)
cv2.circle(raw,left, 5,(0,255,255), 3);

cv2.putText(raw, "Center", (center[0] - 10, center[1] - 20), 0,0.8, (0,255,255),2)
cv2.circle(raw,center, 5,(0,255,255), 3);


cv2.putText(raw, "Right", (right[0] - 10, right[1] - 20), 0,0.8, (0,255,255),2)
cv2.circle(raw,right, 5,(0,255,255), 3);


#Show Images

cv2.imshow("Raw", raw)
cv2.imshow("Mask", mask)
while(True):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
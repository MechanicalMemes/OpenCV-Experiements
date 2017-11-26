import cv2
import numpy
import glob

# This is a basic Glyph Detector. Its black magic. Much of this I wrote at 2AM and don't fully remeber what I did,
# but it some how "works". So Enjoy! Keep in mind this will be ported to an Android Library shortly. This is mearly
# for easy exerpiementing. - Alex "Phil" Carter from FTC Team 7195,6179 and Disnode Team :)

# SCORING
# Each parameter returns a normalized float between 0-1
# We then calculate the punishment. If 1.0 is bad (in the case of Distance, the further away an object
# is the worst it should be weighted) we subtract this punishment from 1, resulting in 0. Multiply this by
# the weight so we can tune and multiply this to the score.
# If 1.0 is good we just directly weigh it and multiply.

# Settings
imagePath = "./images/jewels"  # Path to Images
imageSize = (800, 600)  # Resize Images to this

debug_show_preprocessed = False  # Show the PreProcessed Image
debug_show_filtered = False  # Show the Filtered Image
debug_show_blue_result = True
debug_draw_stats = True  # Show Stats for Each Rectangle (Very Spammy)
debug_draw_rects = True  # Draw all found rectables


def preprocess_image(input_image):
    return cv2.resize(input_image, imageSize)


def filter_blue(input_image):
    hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

    lower = numpy.array([80, 55, 40])
    upper = numpy.array([115, 255, 230])

    mask = cv2.inRange(hsv, lower, upper)
    kernel = numpy.ones((5, 5), numpy.uint8)
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)

    if debug_show_filtered:
        cv2.imshow("Mask Blue", mask)

    return mask


def detect_blue(input_image):
    processed = preprocess_image(input_image)
    mask = filter_blue(processed)

    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    foundObjects = []
    for c in contours:
        cnt = c
        if cv2.contourArea(c) >= 500:
            x, y, w, h = cv2.boundingRect(cnt)
            if 1 - abs(w / h) < 0.2:

                if debug_draw_rects:
                    cv2.rectangle(processed, (x, y), (x + w, y + h), (255, 255, 255), 2)
                if debug_draw_stats:
                    cv2.putText(processed, "Cube " + str(x), (x + 20, y + 30), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                (255, 255, 255))

                foundObjects.append((x, y, w, h))

    currentMax = -1
    chosenObject = (-1, -1, -1, -1)
    for obj in foundObjects:
        area = obj[0] * obj[1]
        if area > currentMax:
            currentMax = area
            chosenObject = obj

    cv2.rectangle(processed, (chosenObject[0], chosenObject[1]),
                  (chosenObject[0] + chosenObject[2], chosenObject[1] + chosenObject[3]), (255, 0, 0), 2)
    if debug_show_blue_result:
        cv2.imshow("Blue Result", processed)
    return chosenObject[0]


def DetectRed(input_image):
    hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

    lowerMask = cv2.inRange(hsv, numpy.array([0, 150, 100]), numpy.array([10, 255, 255]))
    upperMask = cv2.inRange(hsv, numpy.array([160, 100, 100]), numpy.array([179, 255, 255]))
    # mask = cv2.bilateralFilter(mask, 11, 17, 17)
    # mask = cv2.blur(mask, (10,10))

    mask = cv2.addWeighted(lowerMask, 1.0, upperMask, 1.0, 0.0)
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    foundObjects = []
    for c in contours:
        cnt = c
        if cv2.contourArea(c) >= 500:
            x, y, w, h = cv2.boundingRect(cnt)
            if 1 - numpy.abs(w / h) < 0.1:
                cv2.putText(input_image, "Cube" + str(x), (x + 20, y + int(h / 2)), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                            (0, 0, 255))
                cv2.rectangle(input_image, (x, y), (x + w, y + h), (255, 255, 255), 2)
                foundObjects.append((x, y, w, h))

    currentMax = -1
    chosenRedObject = (-1, -1, -1, -1)
    for obj in foundObjects:
        area = obj[0] * obj[1]
        if area > currentMax:
            currentMax = area
            print("New Max Red Area: " + str(area))
            chosenRedObject = obj

    cv2.rectangle(input_image, (chosenRedObject[0], chosenRedObject[1]),
                  (chosenRedObject[0] + chosenRedObject[2], chosenRedObject[1] + chosenRedObject[3]), (0, 0, 255), 2)

    cv2.imshow("Raw Red", input_image)
    cv2.imshow("mask Red", mask)
    return chosenRedObject[0]


# Load Images and Run
print("Starting Jewel Detector")
files = glob.glob(imagePath + "/*.jpg")

for file in files:
    print("Running file: " + file)
    image = cv2.imread(file)
    blue = detect_blue(image)
    # red = DetectRed(image)
    # print("Red" + str(red))
    # print("Blue" + str(blue))
    # final = image

    # if blue < red:
    #    cv2.putText(final, "Blue - Red", (20, 50), 0, 1.0, (0, 255, 0), 2)
    # if red < blue:
    #    cv2.putText(final, "Red - Blue", (20, 50), 0, 1.0, (0, 255, 0), 2)
    # cv2.imshow("Final", final)
    while (True):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

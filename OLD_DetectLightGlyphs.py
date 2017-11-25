import cv2
import numpy


def Process(raw):
    raw = cv2.resize(raw, (1280, 720))
    height, width = raw.shape[:2]
    grey = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (3, 3), 1)
    grey = cv2.bilateralFilter(grey, 11, 17, 17)
    edges = cv2.Canny(grey, 20, 50)
    structure = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 45))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, structure)

    img1, cnts, hq = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = None

    cubeMask = numpy.zeros((height, width), numpy.uint8)

    chosenRect = None
    chosenDistance = 10000
    for c in cnts:
        # approximate the contour


        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        hullPoints = cv2.convexHull(c)
        cv2.polylines(raw, hullPoints, True, (0, 255, 255), 4)
        rect = cv2.boundingRect(c)

        x, y, w, h = rect
        if cv2.contourArea(c) >= 4000:
            if len(approx) <= 6 and len(approx) >= 4:

                if y > (height / 4):
                    cv2.drawContours(raw, [approx], -1, (255, 255, 0), 3)

                    center = rect[0] + (rect[2] / 2)
                    distance = (center - (width / 2))
                    distance = abs(distance)
                    cv2.circle(raw, (int(center), int(rect[1] + (rect[3] / 2))), 4, (0, 255, 255), 3)
                    cv2.putText(raw, str(distance), (int(center), int(rect[1] + (rect[3] / 2))), 1, 1.5,
                                (255, 255, 255), 2)

                    print("Distance: " + str(distance) + " vs " + str(chosenDistance))

                    if distance < chosenDistance:
                        print("CHOSEN!")
                        chosenRect = rect
                        chosenDistance = distance


    (x, y, w, h) = chosenRect

    result = int(x + (w / 2))

    cv2.rectangle(raw, (x, y), (x + w, y + h), (0, 255, 0), 5)
    cv2.putText(raw, "Chosen", (x + 10, y + int(h / 2) - 40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(raw, "RESULT: " + str(result), (result, 30), 0, 1, (0, 255, 0), 1)
    cv2.line(raw, (result, 0), (result, 1000), (0, 255, 0), 3)

    cv2.imshow("Raw", raw)

    cv2.imshow("Edges", edges)

    while (True):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


import glob

for path in glob.glob("./images/glyphs/*.jpg"):
    Process(cv2.imread(path))

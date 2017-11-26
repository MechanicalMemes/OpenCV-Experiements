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
imagePath = "./images/glyphs/robot_level"  # Path to Images
imageSize = (1280, 720)  # Resize Images to this

debug_show_preprocessed = False  # Show the PreProcessed Image
debug_show_filtered = False  # Show the Filtered Image
debug_draw_stats = True  # Show Stats for Each Rectangle (Very Spammy)
debug_draw_center = False  # Draw Center Line on the screen
debug_draw_rects = True  # Draw all found rectables

# Weights for scoring
score_ratio_weight = 0.5
score_distance_x_weight = 0.8
score_distance_y_weight = 1
score_area_weight = 0.4


# Process and Find Glyphs

def process_image(input_mat):
    output = input_mat
    output = cv2.resize(output, imageSize)
    processed = preprocess(input_mat)
    filtered = apply_filters(processed)

    countor_image, cnts, hq = cv2.findContours(filtered.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    cnts.pop(0)  # Remove First Index which is always the entire screen

    detected_rects = []
    chosen_rect = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        # approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        rect = cv2.boundingRect(c)
        x, y, w, h = rect

        center_point = (int(x + (w / 2)), int(y + (h / 2)))
        cube_ratio = max(abs(h / w), abs(w / h))

        score = 100

        # Ratio score calculation (1.0 is bad)
        distance_from_perfect = abs(1 - cube_ratio)
        score_ratio_punishment = 1 - distance_from_perfect
        score_ratio = score_ratio_punishment * score_ratio_weight
        score = score * score_ratio

        # Position score (1.0 is bad)
        distance_from_center_x = (imageSize[0] / 2) - center_point[0]
        distance_from_center_y = (imageSize[1]) - center_point[1]
        distance_from_center_x = distance_from_center_x
        distance_from_center_y = distance_from_center_y
        distance_from_center_x_normalized = abs(distance_from_center_x / imageSize[0])
        distance_from_center_y_normalized = abs(distance_from_center_y / imageSize[1])

        score_distance_x_punishment = 1 - distance_from_center_x_normalized
        score_distance_x = score_distance_x_punishment * score_distance_x_weight
        score_distance_y_punishment = 1 - distance_from_center_y_normalized
        score_distance_y = score_distance_y_punishment * score_distance_y_weight

        score = score * score_distance_x
        score = score * score_distance_y


        # Area Scoring (1.0 is good)
        areas = [cv2.contourArea(c) for c in cnts]
        max_index = numpy.argmax(areas)
        min_index = numpy.argmin(areas)

        max_area = areas[max_index]
        min_area = areas[min_index]

        area = cv2.contourArea(c)
        area_normalized = (area / min_area) / (max_area / min_area)
        score_area_punishment = area_normalized
        score_area = score_area_punishment * score_distance_x_weight
        score = score * score_area

        # Choose Rect based on score

        detected_rects.append((x, y, w, h, score))

        if chosen_rect is None:
            chosen_rect = (x, y, w, h, score)

        if chosen_rect[4] < score:
            chosen_rect = (x, y, w, h, score)

        # Debug

        if debug_draw_rects:
            cv2.circle(output, center_point, 1, (0, 255, 255), 4)
            cv2.rectangle(output, (x, y), ((x + w), (y + h)), (255, 255, 9), 2)
        if debug_draw_stats:
            string_ratio = "Ratio %4.3f" % cube_ratio
            string_area = "Area %4.3f" % area
            string_distance = "Distance %4.3f / %4.3f" % (distance_from_center_x_normalized , distance_from_center_y_normalized)
            string_area_score = "AScore %4.3f" % score_area
            string_distance_score = "DScore %4.2f / %4.2f" % (score_distance_x, score_distance_y)
            string_ratio_score = "RScore %4.3f" % score_ratio
            string_score = "Score %4.3f" % score

            cv2.putText(output, string_ratio, (x + 5, y + 15), 0, 0.35, (0, 255, 255), 1)
            cv2.putText(output, string_area, (x + 5, y + 30), 0, 0.35, (0, 255, 255), 1)
            cv2.putText(output, string_distance, (x + 5, y + 45), 0, 0.35, (0, 255, 255), 1)
            cv2.putText(output, string_area_score, (x + 5, y + 60), 0, 0.35, (0, 255, 255), 1)
            cv2.putText(output, string_distance_score, (x + 5, y + 75), 0, 0.35, (0, 255, 255), 1)
            cv2.putText(output, string_ratio_score, (x + 5, y + 90), 0, 0.35, (0, 255, 255), 1)
            cv2.putText(output, string_score, (x + 5, y + 115), 0, 0.5, (255, 255, 0), 1)

    print(chosen_rect)
    x, y, w, h, score = chosen_rect
    cv2.rectangle(output, (x, y), ((x + w), (y + h)), (0, 255, 0), 2)

    chosen_center = (x + (w / 2), y + (h / 2))
    distance = chosen_center[0] - (imageSize[0] / 2)

    if debug_draw_center:
        cv2.line(output, (int((imageSize[0] / 2)), 0), (int((imageSize[0] / 2)), imageSize[1]), (255, 255, 255), 5,
                 lineType=4)

    cv2.putText(output, "Chosen %3.2f" % distance, (x, (y - 5)), 0, 0.7, (0, 255, 100), 2)

    cv2.imshow("Output", output)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Resizes and Convert Colors. Preparing the image for use
def preprocess(input_mat):
    resized = cv2.resize(input_mat, imageSize)
    grey = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # grey = resized

    if debug_show_preprocessed:
        cv2.imshow("PreProcessed", grey)

    return grey


# Applyes Filters to the images
def apply_filters(input_mat):
    blurred = cv2.GaussianBlur(input_mat, (3, 3), 1)  # Jack-ify the image
    blurred = cv2.bilateralFilter(blurred, 11, 17, 17)
    edges = cv2.Canny(blurred, 20, 50)
    structure = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 45))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, structure)

    if debug_show_filtered:
        cv2.imshow("Filtered", edges)

    return edges


# Load Images and Run

files = glob.glob(imagePath + "/*.jpg")

for file in files:
    print("Running file: " + file)
    image = cv2.imread(file)
    process_image(image)

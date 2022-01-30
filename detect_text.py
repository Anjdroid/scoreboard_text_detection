import time
from tracemalloc import start
import cv2 as cv
import numpy as np
import pytesseract
from pytesseract import Output
from imutils.object_detection import non_max_suppression
from dataset import VideoDataset
import re


VIDEO_FILE = 'data/top-100-shots-rallies-2018-atp-season.mp4'
JSON_FILE = 'data/top-100-shots-rallies-2018-atp-season-scoreboard-annotations.json'
DATA_FOLDER = 'data'
MODEL_PATH = 'model/seg-unet.pth'
EAST_MODEL_PATH = 'model/frozen_east_text_detection.pb'
MIN_CONFIDENCE = 0.0001
MIN_CONF_PY = 10
letters_pattern = '[»A-Za-z]'

vdata = VideoDataset(VIDEO_FILE, JSON_FILE)
keys = list(vdata.videos.keys())

def detect_chars(sb):
    detected_chars = []
    h, w, _ = sb.shape
    img = sb.copy()
    # detect characters on scoreboard
    boxes = pytesseract.image_to_boxes(img) 
    for b in boxes.splitlines():
        b = b.split(' ')
        # match detected characters to letters
        if re.match(letters_pattern, b[0]):
            img = cv.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
            detected_chars.append(b[0])
    #cv.imshow('CHARACTER DETECTION', img)
    #cv.waitKey(0)
    return detect_chars, img


def detect_words(sb):
    img = sb.copy()
    detected_words = []
    # detect words on scoreboard
    words = pytesseract.image_to_data(img, output_type=Output.DICT)
    n_boxes = len(words['text'])
    for i in range(n_boxes):
        # check detected word confidence
        if int(float(words['conf'][i])) > MIN_CONF_PY:
            print('==DETECTED TEXT==', words['text'][i])
            print('==CONFIDENCE SCORE==', words['conf'][i])
            # match detected text to pattern
            if re.match(letters_pattern, words['text'][i]):
                # draw words bboxes
                (x, y, w, h) = (words['left'][i], words['top'][i], words['width'][i], words['height'][i])
                img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                detected_words.append(words['text'][i])
    #cv.imshow('WORD DETECTION', img)
    #cv.waitKey(0)
    return detected_words, img


def calc_bbox(x, y, angles, x_0, x_1, x_2, x_3):
    # get bbox offset vectors
    (offset_x, offset_y) = (x * 4.0, y * 4.0)
    # text rotation
    angle = angles[x]
    cos = np.cos(angle)
    sin = np.sin(angle)
    # width & height from bbox
    h = x_0[x] + x_2[x]
    w = x_1[x] + x_3[x]
    # start & end x,y
    end_x = int(offset_x + (cos * x_1[x]) + (sin * x_2[x]))
    end_y = int(offset_y - (sin * x_1[x]) + (cos * x_2[x]))
    start_x = int(end_x - w)
    start_y = int(end_y - h)
    return (start_x, start_y, end_x, end_y)


def detect_text_EAST(image):
    # output layer names for the EAST detector model 
    # first is the output probabilities 
    # second for bbox detection
    layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]
    orig = image.copy()
    img = (orig.copy()).astype('uint8')
    W, H = orig.shape[:2]
    # load the pre-trained EAST text detector
    print('==Loading EAST text detector==')
    net = cv.dnn.readNet(EAST_MODEL_PATH)
    #print(img.shape, type(img), img.dtype)
    # construct a blob from frame
    blob = cv.dnn.blobFromImage(img, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    # get model predictions: probabilities, bboxes
    (scores, bbox) = net.forward(layerNames)
    end = time.time()
    print('==EAST DETECTION TIME==\n', (end - start))

    (num_rows, num_cols) = scores.shape[2:4]
    bboxes = []
    probabilities = []
    # loop over rows
    for y in range(0, num_rows):
        # get score confidence
        probability = scores[0, 0, y]
        # get bbox geometry
        x_0 = bbox[0, 0, y]
        x_1 = bbox[0, 1, y]
        x_2 = bbox[0, 2, y]
        x_3 = bbox[0, 3, y]
        angles = bbox[0, 4, y]
    	# loop over rows
        for x in range(0, num_cols):
            # check detection confidence
            if probability[x] < MIN_CONFIDENCE:
                continue
            predicted_bbox = calc_bbox(x, y, angles, x_0, x_1, x_2, x_3)
            bboxes.append(predicted_bbox)
            probabilities.append(probability[x])
    # apply NMS, remove overlapping regions
    better_boxes = non_max_suppression(np.array(bboxes), probs=probabilities)
    # loop over the bounding boxes
    #better_boxes = bboxes
    for (startX, startY, endX, endY) in better_boxes:
        # draw the bounding box on the image
        cv.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
    # show the output image
    #cv.imshow("TEXT DETECTION", img)
    #cv.waitKey(0)
    return img

def run_text_detection():
    # iterate over frames
    for key in keys:
        print(key)
        frame, mask, sb, gt = vdata[key]

        # TEXT DETECTION with Pytasseract
        chars, i1 = detect_chars(sb)
        words, i2 = detect_words(sb)
        cv.imwrite('results/result_char_det_tess_' + key + '.png', i1.astype(np.uint8))
        cv.imwrite('results/result_word_det_tess_' + key + '.png', i2.astype(np.uint8))
        
        if len(words) < 2:
            print("=detected players= ", words)
            print("==FAIL: only 1 player detected")
        elif len(words) == 2:        
            p1 = words[0]
            p2 = words[0]

            if p1[0] == '»':
                p1 = p1.strip('»')
                print('==STARTING PLAYER== \n', p1)
            else:
                p2 = p2.strip('»')
                print('==STARTING PLAYER== \n', p2)
            print('==DETECTED PLAYERS==\n', p1, p2)
        else:
            print("=detected players= ", words)
            print("==FAIL: more than 2 players detected")

        # TEXT detection with EAST
        img = detect_text_EAST(sb)
        cv.imwrite('results/result_text_det_east_' + key + '.png', img.astype(np.uint8))



if __name__ == "__main__":
    run_text_detection()

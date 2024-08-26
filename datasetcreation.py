import numpy as np
import cv2 as cv
import tensorflow as tf
import keras
import handtrackingmodule as htm
import os
import time

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
wCam, hCam = 648, 488

capture = cv.VideoCapture(0)
capture.set(3, wCam)
capture.set(4, hCam)

folder = "Data/1.2"
if not os.path.exists(folder):
    os.makedirs(folder)

detector = htm.handDetector(detectCon=0.7)

while True:
    isTrue, frame = capture.read()
    if not isTrue:
        print("End of video or error reading frame.")
        break

    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)

    if lmList:
        x_min = min([lm[1] for lm in lmList])
        x_max = max([lm[1] for lm in lmList])
        y_min = min([lm[2] for lm in lmList])
        y_max = max([lm[2] for lm in lmList])

        padding = 20
        x_min = max(x_min - padding, 0)
        y_min = max(y_min - padding, 0)
        x_max = min(x_max + padding, frame.shape[1])
        y_max = min(y_max + padding, frame.shape[0])

        cropped_frame = frame[y_min:y_max, x_min:x_max]

        cv.imshow('Cropped Frame', cropped_frame)

    cv.imshow('Video', frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord('s'):
        timestamp = int(time.time())
        filename = f'{folder}/frame_{timestamp}.jpg'
        cv.imwrite(filename, cropped_frame)
        print(f"Image saved: {filename}")
    elif key == ord('d'):
        break

capture.release()
cv.destroyAllWindows()

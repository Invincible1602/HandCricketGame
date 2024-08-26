import numpy as np
import cv2 as cv
import tensorflow as tf

import handtrackingmodule as htm
import os
from PIL import Image, ImageOps

# Disable OneDNN optimizations for potential performance improvements
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the model
model = tf.keras.models.load_model('keras_model1.h5')

# Load the labels
with open('labels1.txt', 'r') as f:
    class_names = f.readlines()

# Camera dimensions
wCam, hCam = 648, 488

# Start video capture
capture = cv.VideoCapture(0)
capture.set(3, wCam)
capture.set(4, hCam)

# Initialize hand detector
detector = htm.handDetector(detectCon=0.9)

# Prepare an array to hold the preprocessed image data
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

while True:
    isTrue, frame = capture.read()
    if not isTrue:
        print("End of video or error reading frame.")
        break

    # Detect hands and get landmarks
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)

    if lmList:
        # Find the bounding box for the hand
        x_min = min([lm[1] for lm in lmList])
        x_max = max([lm[1] for lm in lmList])
        y_min = min([lm[2] for lm in lmList])
        y_max = max([lm[2] for lm in lmList])

        # Apply padding
        padding = 20
        x_min = max(x_min - padding, 0)
        y_min = max(y_min - padding, 0)
        x_max = min(x_max + padding, frame.shape[1])
        y_max = min(y_max + padding, frame.shape[0])

        # Crop and preprocess the frame
        cropped_frame = frame[y_min:y_max, x_min:x_max]
        if cropped_frame.size > 0:
            image = Image.fromarray(cropped_frame)
            image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            data[0] = normalized_image_array

            # Predict the gesture
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index].strip()
            confidence_score = prediction[0][index]

            # Display results on the frame
            cv.putText(frame, f"Gesture: {class_name}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
            cv.putText(frame, f"Confidence: {confidence_score:.2f}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

    # Show the frame
    cv.imshow('Video', frame)

    # Exit on 'd' key press
    if cv.waitKey(1) & 0xFF == ord('d'):
        break

# Release resources
capture.release()
cv.destroyAllWindows()

import numpy as np
import cv2 as cv
import tensorflow as tf
import handtrackingmodule as htm
import os
from PIL import Image, ImageOps
import random
import time

# Disable OneDNN optimizations for potential performance improvements
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the trained CNN model
try:
    model = tf.keras.models.load_model('keras_model1.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Load the labels from the file
try:
    with open('labels1.txt', 'r') as f:
        class_names = f.read().splitlines()
except Exception as e:
    print(f"Error loading labels: {e}")
    exit(1)

# Camera dimensions
wCam, hCam = 648, 488

# Start video capture
capture = cv.VideoCapture(0)
if not capture.isOpened():
    print("Error: Could not open camera.")
    exit(1)

capture.set(3, wCam)
capture.set(4, hCam)

# Initialize hand detector
try:
    detector = htm.handDetector(detectCon=0.9)
except Exception as e:
    print(f"Error initializing hand detector: {e}")
    capture.release()
    exit(1)

# Prepare an array to hold the preprocessed image data
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Load images for random numbers
number_images = {}
for num in [1, 2, 3, 4, 6]:
    img = cv.imread(f'num_images/{num}.png')
    if img is not None:
        img = cv.resize(img, (100, 100))  # Resize to fit the display area
        number_images[num] = img
    else:
        print(f"Warning: Image for number {num} not found.")

player_score = 0
computer_score = 0
wickets = 0
computer_wickets = 0  # Initialize computer wickets
game_over = False
waiting_for_input = False
show_intro = True
random_number_image = None
random_number = None
current_round = 0
max_rounds = 10
player_batting = True

def generate_random_number():
    numbers = [1, 2, 3, 4, 6]
    weights = [1, 1, 2, 3, 5]  # Higher weight for the number 6
    return random.choices(numbers, weights=weights, k=1)[0]

def display_text(frame, text, position, color, scale=0.8, thickness=1, bg_color=None):
    if bg_color:
        (text_width, text_height), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x, y = position
        cv.rectangle(frame, (x - 10, y - text_height - 10), (x + text_width + 10, y + 10), bg_color, -1)
    cv.putText(frame, text, position, cv.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv.LINE_AA)

def display_intro(frame):
    display_text(frame, "Welcome to Hand Cricket!", (50, 100), (234, 255, 255), scale=1.0)
    display_text(frame, "Press 's' to Start", (50, 200), (0, 255, 0), scale=1.2, thickness=2, bg_color=(0, 0, 0))

def display_game_status(frame):
    display_text(frame, f"Player Score: {player_score}", (50, 50), (0, 255, 0), scale=0.8)
    display_text(frame, f"Computer Score: {computer_score}", (50, 100), (0, 255, 0), scale=0.8)
    display_text(frame, f"Player Wickets: {wickets}", (50, 150), (0, 255, 0), scale=0.8)
    display_text(frame, f"Computer Wickets: {computer_wickets}", (50, 200), (0, 255, 0), scale=0.8)  # Display computer wickets
    display_text(frame, f"Round: {current_round}/{max_rounds}", (50, 250), (0, 255, 0), scale=0.8)
    display_text(frame, "Press 's' to Continue, 'n' to Restart, or 'q' to Quit", (50, 300), (0, 255, 0), scale=0.8)

    if random_number_image is not None:
        y_offset = 350
        x_offset = 50
        frame[y_offset:y_offset + random_number_image.shape[0],
              x_offset:x_offset + random_number_image.shape[1]] = random_number_image
    else:
        display_text(frame, "No image available for the number", (50, 300), (0, 0, 255), scale=0.8)

def check_game_over():
    global game_over
    if wickets >= 3 or computer_wickets >= 3 or current_round >= max_rounds:  # Include computer wickets in the game over condition
        game_over = True
        return True
    return False

try:
    while True:
        isTrue, frame = capture.read()
        if not isTrue:
            print("End of video or error reading frame.")
            break

        if show_intro:
            display_intro(frame)
            cv.imshow('Hand Cricket Game', frame)
            key = cv.waitKey(1) & 0xFF
            if key == ord('s'):
                show_intro = False
                current_round = 1  # Start from round 1
                random_number = generate_random_number()  # Initialize the random number
                player_batting = True  # Player starts batting
                computer_score = 0
                player_score = 0
                wickets = 0
                computer_wickets = 0  # Reset computer wickets
                continue
            elif key == ord('q'):
                break
            continue

        if game_over:
            display_game_status(frame)
            cv.imshow('Hand Cricket Game', frame)
            key = cv.waitKey(1) & 0xFF
            if key == ord('n'):
                player_score = 0
                computer_score = 0
                wickets = 0
                computer_wickets = 0  # Reset computer wickets
                game_over = False
                waiting_for_input = False
                show_intro = True
            elif key == ord('q'):
                break
            continue

        if waiting_for_input:
            display_game_status(frame)
            cv.imshow('Hand Cricket Game', frame)
            key = cv.waitKey(1) & 0xFF
            if key == ord('s'):
                if current_round < max_rounds:
                    current_round += 1
                    waiting_for_input = False
                else:
                    game_over = True
                continue
            elif key == ord('n'):
                player_score = 0
                computer_score = 0
                current_round = 0
                game_over = False
                waiting_for_input = False
                show_intro = True
            elif key == ord('q'):
                break
            continue

        # Detecting gesture
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
            if cropped_frame.size > 0:
                try:
                    # Display the frame and wait before processing
                    cv.imshow('Hand Cricket Game', frame)
                    cv.waitKey(1500)  # Increased delay before prediction

                    image = Image.fromarray(cropped_frame)
                    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
                    image_array = np.asarray(image)
                    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
                    data[0] = normalized_image_array

                    # Add delay before model prediction to ensure it is not too frequent
                    cv.waitKey(500)  # 500ms delay before prediction

                    prediction = model.predict(data)
                    predicted_class = class_names[np.argmax(prediction)]

                    try:
                        predicted_number = int(predicted_class)
                    except ValueError:
                        predicted_number = -1

                    random_number = generate_random_number()
                    random_number_image = number_images.get(random_number)

                    if player_batting:
                        if predicted_number == random_number:
                            wickets += 1
                            if check_game_over():
                                game_over = True
                        else:
                            player_score += predicted_number
                        player_batting = not player_batting
                    else:
                        if predicted_number == random_number:
                            computer_wickets += 1  # Update computer wickets
                            if check_game_over():
                                game_over = True
                        else:
                            computer_score += random_number
                        player_batting = not player_batting

                    waiting_for_input = True

                except Exception as e:
                    print(f"Error during prediction or game logic: {e}")

        display_game_status(frame)
        cv.imshow('Hand Cricket Game', frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break

except KeyboardInterrupt:
    print("Game interrupted by user.")
finally:
    capture.release()
    cv.destroyAllWindows()

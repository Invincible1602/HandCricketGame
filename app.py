import tkinter as tk
from tkinter import messagebox
import cv2 as cv
import numpy as np
import tensorflow as tf
import handtrackingmodule as htm
import os
from PIL import Image, ImageOps
import random


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


try:
    model = tf.keras.models.load_model('keras_model1.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

try:
    with open('labels1.txt', 'r') as f:
        class_names = f.read().splitlines()
except Exception as e:
    print(f"Error loading labels: {e}")
    exit(1)


def play_game():
    
    wCam, hCam = 648, 488

    
    capture = cv.VideoCapture(0)
    if not capture.isOpened():
        print("Error: Could not open camera.")
        return

    capture.set(3, wCam)
    capture.set(4, hCam)

    #
    try:
        detector = htm.handDetector(detectCon=0.9)
    except Exception as e:
        print(f"Error initializing hand detector: {e}")
        capture.release()
        return


    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


    number_images = {}
    for num in [1, 2, 3, 4, 6]:
        img = cv.imread(f'num_images/{num}.png')
        if img is not None:
            img = cv.resize(img, (100, 100))  
            number_images[num] = img
        else:
            print(f"Warning: Image for number {num} not found.")


    score = [0, 0]
    wickets = [0, 0]
    current_player = 0
    game_over = False
    waiting_for_input = False
    show_intro = True
    random_number_image = None
    random_number = None
    current_round = 0
    max_rounds = 10

    def generate_random_number():
        numbers = [1, 2, 3, 4, 6]
        weights = [1, 1, 2, 3, 5]
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
        display_text(frame, f"Player 1 Score: {score[0]}", (50, 50), (0, 255, 0), scale=0.8)
        display_text(frame, f"Player 1 Wickets: {wickets[0]}", (50, 100), (0, 255, 0), scale=0.8)
        display_text(frame, f"Player 2 Score: {score[1]}", (50, 150), (0, 255, 0), scale=0.8)
        display_text(frame, f"Player 2 Wickets: {wickets[1]}", (50, 200), (0, 255, 0), scale=0.8)
        display_text(frame, f"Round: {current_round}/{max_rounds}", (50, 250), (0, 255, 0), scale=0.8)
        display_text(frame, f"Player {current_player + 1}'s Turn", (50, 300), (0, 255, 255), scale=0.8)
        display_text(frame, "Press 's' to Continue, 'n' to Restart, or 'q' to Quit", (50, 350), (0, 255, 0), scale=0.8)

        if random_number_image is not None:
            y_offset = 400
            x_offset = 50
            resized_image = cv.resize(random_number_image, (100, 80))
            frame[y_offset:y_offset + resized_image.shape[0],
            x_offset:x_offset + resized_image.shape[1]] = resized_image
        else:
            display_text(frame, "No image available for the number", (50, 450), (0, 0, 255), scale=0.8)

    def display_result(frame):
        if score[0] > score[1]:
            winner = "Player 1 Wins!"
        elif score[1] > score[0]:
            winner = "Player 2 Wins!"
        else:
            winner = "It's a Draw!"

        display_text(frame, f"Final Score - Player 1: {score[0]}", (50, 50), (0, 255, 0), scale=1.0)
        display_text(frame, f"Final Score - Player 2: {score[1]}", (50, 100), (0, 255, 0), scale=1.0)
        display_text(frame, winner, (50, 200), (0, 0, 255), scale=1.2, thickness=2, bg_color=(0, 0, 0))
        display_text(frame, "Press 'n' to Restart or 'q' to Quit", (50, 300), (0, 255, 0), scale=0.8)

    def show_game_result_ui():
        result_window = tk.Tk()
        result_window.title("Game Over")
        result_window.geometry("400x300")

        if score[0] > score[1]:
            winner_message = f"Player 1 Wins with a score of {score[0]}!"
        elif score[1] > score[0]:
            winner_message = f"Player 2 Wins with a score of {score[1]}!"
        else:
            winner_message = "It's a Draw!"

        result_label = tk.Label(result_window, text=winner_message, padx=20, pady=20, justify=tk.CENTER, font=("Arial", 16))
        result_label.pack()

        def restart_game():
            result_window.destroy()
            play_game()

        def quit_game():
            result_window.destroy()
            root.quit()

        restart_button = tk.Button(result_window, text="Restart Game", command=restart_game)
        restart_button.pack(pady=10)

        quit_button = tk.Button(result_window, text="Quit", command=quit_game)
        quit_button.pack(pady=10)

        result_window.mainloop()

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
                    current_round = 1
                    random_number = generate_random_number()
                    continue
                elif key == ord('q'):
                    break
                continue

            if game_over:
                display_result(frame)
                cv.imshow('Hand Cricket Game', frame)
                key = cv.waitKey(1) & 0xFF
                if key == ord('n'):
                    score = [0, 0]
                    wickets = [0, 0]
                    current_player = 0
                    game_over = False
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
                        current_player = 1 - current_player
                        waiting_for_input = False
                    else:
                        game_over = True
                        show_game_result_ui()
                    continue
                elif key == ord('n'):
                    score = [0, 0]
                    wickets = [0, 0]
                    current_player = 0
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
                        cv.imshow('Hand Cricket Game', frame)
                        cv.waitKey(1500)

                        image = Image.fromarray(cropped_frame)
                        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
                        image_array = np.asarray(image)
                        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
                        data[0] = normalized_image_array

                        cv.waitKey(500)

                        prediction = model.predict(data)
                        predicted_class = class_names[np.argmax(prediction)]

                        try:
                            predicted_number = int(predicted_class)
                        except ValueError:
                            predicted_number = None

                        if predicted_number is not None:
                            random_number = generate_random_number()
                            random_number_image = number_images.get(random_number)

                            if predicted_number == random_number:
                                wickets[current_player] += 1
                                if wickets[current_player] >= 3:
                                    game_over = True
                            else:
                                score[current_player] += predicted_number

                            waiting_for_input = True

                    except Exception as e:
                        print(f"Error processing frame: {e}")

            display_game_status(frame)
            cv.imshow('Hand Cricket Game', frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error during game loop: {e}")

    finally:
        capture.release()
        cv.destroyAllWindows()



def start_game():
    play_game()



def start_game():
    root.destroy()
    play_game()

def quit_game():
    root.quit()

def show_instructions():
    global root
    root = tk.Tk()
    root.title("Hand Cricket Instructions")
    root.geometry("700x600")


    root.configure(bg="#ff0000")

    instructions = """Welcome to Hand Cricket!

    Rules:
    - Use hand gestures to play the game.
    - The game consists of 10 rounds.
    - Each player will play against the computer turn by turn.
    - The computer will randomly generate a number between 1 and 6.
    - The player must predict the number using hand gestures.
    - If the player's prediction matches the computer's number, it's a wicket.
    - If the prediction is different, the player scores the predicted number.
    - The player with the highest score at the end wins.
    - Follow the on-screen instructions to continue.

    Press 's' to Start the Game
    Press 'q' to Quit
    """


    label = tk.Label(root, text=instructions, padx=20, pady=20, justify=tk.LEFT, font=("Arial", 14), bg="#00ff00")
    label.pack()


    start_button = tk.Button(root, text="Player Vs Player", command=start_game, font=("Arial", 14), bg="#4CAF50", fg="white")
    start_button.pack(pady=20)

    quit_button = tk.Button(root, text="Quit", command=quit_game, font=("Arial", 14), bg="#f44336", fg="white")
    quit_button.pack(pady=10)

    root.mainloop()

show_instructions()

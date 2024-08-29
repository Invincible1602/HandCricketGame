# HandCricketGame

The Hand Cricket game is a Python project that combines Tkinter for the user interface and OpenCV for hand gesture detection. The game mimics traditional cricket using hand gestures, where players score runs or take wickets based on matching gestures with randomly generated numbers by the computer. It supports both single-player and two-player modes, allowing players to compete against each other or the computer. 

The keras model is present in my google drive and it is not uploding in github: https://drive.google.com/file/d/110ova9ywFprRDh2X5m8MC3SPv31nliXo/view?usp=drive_link

In single-player mode, 
The game consists of up to 10 rounds, alternating between the player batting and the computer bowling. The game can switch roles, where the player also bowls, and the computer bats.
The game begins with the player batting, trying to score as many runs as possible before getting out (3 wickets).
After each gesture is detected, the game compares the player's predicted number with the computer's random number.
If the numbers match, a wicket is taken. If not, the score is updated.
After 10 rounds or when 3 wickets are taken, the game ends, and the player is shown their final score.

In two-player mode,
The computer randomly generates a number between 1 and 6.
The 2 players play turn-wise against the computer, must predict the number using specific hand gestures, which are detected by the camera.
If the player's prediction matches the computer's number, the player loses a wicket.
If the prediction is different, the player scores the predicted number.
The game ends after 10 rounds or when the player loses all wickets.
The player with the highest score at the end of the game wins.

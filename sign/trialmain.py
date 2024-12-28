import pickle 
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import tkinter as tk
from tkinter import StringVar, Label, Button, Frame
from PIL import Image, ImageTk
import threading
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=1)

# Text-to-Speech setup
engine = pyttsx3.init()

# label mapping
labels_dict = {
    0: 'Hello', 1: 'Bye', 2: 'Superb!', 3: 'Sad', 4: 'I love you!', 5: 'Namaste!',
    6: 'See you!', 7: 'Best', 8: 'Not bad', 9: 'J'
}
expected_features = 42

# Initialize buffers and history
stabilization_buffer = []
stable_char = None
word_buffer = ""
sentence = ""

# Speak text in a separate thread
def speak_text(text):
    def tts_thread():
        engine.say(text)
        engine.runAndWait()

    threading.Thread(target=tts_thread, daemon=True).start()


# GUI Setup
root = tk.Tk()
root.title("Sign Language to Text/Speech Conversion")
root.geometry("1300x650")  
root.configure(bg="#000000")  
root.resizable(False, False)  

# Variables for GUI
current_alphabet = StringVar(value="N/A")
current_word = StringVar(value="N/A")
current_sentence = StringVar(value="N/A")
is_paused = StringVar(value="False")
prediction_accuracy = StringVar(value="N/A")

# Title
title_label = Label(root, text="Sign Language to Speech/Text Conversion", font=("Lexend", 28, "bold"), fg="#ffffff", bg="#000000")
title_label.grid(row=0, column=0, columnspan=2, pady=10)

# Layout Frames
video_frame = Frame(root, bg="#161b22", bd=5, relief="solid", width=400, height=400)
video_frame.grid(row=1, column=0, rowspan=3, padx=20, pady=20)
video_frame.grid_propagate(True)

content_frame = Frame(root, bg="#161b22")
content_frame.grid(row=1, column=1, sticky="n", padx=(20, 40), pady=(60, 20))

button_frame = Frame(root, bg="#161b22")
button_frame.grid(row=3, column=1, pady=(10, 20), padx=(10, 20), sticky="n")

# Video feed
video_label = tk.Label(video_frame)
video_label.pack(expand=True)

# Labels
Label(content_frame, text="Current Word/Sign:", font=("Arial", 20), fg="#ffffff", bg="#161b22").pack(anchor="w", pady=(20, 10))
Label(content_frame, textvariable=current_word, font=("Arial", 20), fg="#f39c12", bg="#161b22", wraplength=500, justify="left").pack(anchor="center")

Label(content_frame, text="Prediction Accuracy:", font=("Arial", 20), fg="#ffffff", bg="#161b22").pack(anchor="w", pady=(20, 10))
Label(content_frame, textvariable=prediction_accuracy, font=("Arial", 20), fg="#1abc9c", bg="#161b22", wraplength=500, justify="left").pack(anchor="center")

Label(content_frame, text="Current Sentence:", font=("Arial", 20), fg="#ffffff", bg="#161b22").pack(anchor="w", pady=(20, 10))
Label(content_frame, textvariable=current_sentence, font=("Arial", 20), fg="#9b59b6", bg="#161b22", wraplength=500, justify="left").pack(anchor="center")

# Reset and Pause/Play functions
def reset_sentence():
    global word_buffer, sentence
    word_buffer = ""
    sentence = ""
    current_word.set("N/A")
    current_sentence.set("N/A")
    current_alphabet.set("N/A")

def toggle_pause():
    if is_paused.get() == "False":
        is_paused.set("True")
        pause_button.config(text="Play")
    else:
        is_paused.set("False")
        pause_button.config(text="Pause")

# Buttons
Button(button_frame, text="Reset Sentence", font=("Arial", 16), command=reset_sentence, bg="#21262d", fg="#ffffff", relief="flat", height=2, width=14).grid(row=0, column=0, padx=10)
pause_button = Button(button_frame, text="Pause", font=("Arial", 16), command=toggle_pause, bg="#21262d", fg="#ffffff", relief="flat", height=2, width=12)
pause_button.grid(row=0, column=1, padx=10)
speak_button = Button(button_frame, text="Speak Word", font=("Arial", 16), command=lambda: speak_text(current_word.get()), bg="#21262d", fg="#ffffff", relief="flat", height=2, width=14)
speak_button.grid(row=0, column=2, padx=10)

# Video Capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

# Process frames
def process_frame():
    global stabilization_buffer, stable_char, word_buffer, sentence

    ret, frame = cap.read()
    if not ret:
        return

    if is_paused.get() == "True":
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = img_tk
        video_label.configure(image=img_tk)
        root.after(10, process_frame)
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            if len(data_aux) < expected_features:
                data_aux.extend([0] * (expected_features - len(data_aux)))
            elif len(data_aux) > expected_features:
                data_aux = data_aux[:expected_features]

            probabilities = model.predict_proba([np.asarray(data_aux)])
            max_prob_index = np.argmax(probabilities)
            predicted_character = labels_dict[int(max_prob_index)]
            accuracy_percentage = probabilities[0][max_prob_index] * 100

            prediction_accuracy.set(f"{accuracy_percentage:.2f}%")
            current_word.set(predicted_character)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img_tk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = img_tk
    video_label.configure(image=img_tk)

    root.after(10, process_frame)

process_frame()
root.mainloop()

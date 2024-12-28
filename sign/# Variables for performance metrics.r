# Variables for performance metrics
total_predictions = 0
correct_predictions = 0
false_positives = 0  # Count of incorrect positive predictions

# Function to calculate and update performance metrics
def update_metrics(predicted_character, stable_char):
    global total_predictions, correct_predictions, false_positives
    total_predictions += 1

    if predicted_character == stable_char:
        correct_predictions += 1
    else:
        false_positives += 1

    # Calculate accuracy, precision, and error rate
    accuracy = (correct_predictions / total_predictions) * 100
    precision = (correct_predictions / (correct_predictions + false_positives)) * 100 if correct_predictions + false_positives > 0 else 0
    error_rate = 100 - accuracy

    return round(accuracy, 2), round(precision, 2), round(error_rate, 2)

# In the process_frame function, after predicting the gesture
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # (Existing landmark extraction and prediction code)
        predicted_character = labels_dict[int(prediction[0])]

        # Calculate and update performance metrics
        accuracy, precision, error_rate = update_metrics(predicted_character, stable_char)

        # Stabilization logic (existing code)

        # Draw performance metrics on the video feed
        cv2.putText(frame, f"Accuracy: {accuracy}%", (10, 60), cv2.QT_FONT_NORMAL, 1, (0, 255, 0), 2)  # Green color
        cv2.putText(frame, f"Precision: {precision}%", (10, 90), cv2.QT_FONT_NORMAL, 1, (255, 0, 0), 2)  # Blue color
        cv2.putText(frame, f"Error Rate: {error_rate}%", (10, 120), cv2.QT_FONT_NORMAL, 1, (0, 0, 255), 2)  # Red color

# Display performance metrics in the GUI
Label(content_frame, text="Model Accuracy:", font=("Arial", 20), fg="#ffffff", bg="#161b22").pack(anchor="w", pady=(10, 10))
Label(content_frame, textvariable=StringVar(value=f"{accuracy}%"), font=("Arial", 20), fg="#2ecc71", bg="#161b22").pack(anchor="center")

Label(content_frame, text="Model Precision:", font=("Arial", 20), fg="#ffffff", bg="#161b22").pack(anchor="w", pady=(10, 10))
Label(content_frame, textvariable=StringVar(value=f"{precision}%"), font=("Arial", 20), fg="#3498db", bg="#161b22").pack(anchor="center")

Label(content_frame, text="Model Error Rate:", font=("Arial", 20), fg="#ffffff", bg="#161b22").pack(anchor="w", pady=(10, 10))
Label(content_frame, textvariable=StringVar(value=f"{error_rate}%"), font=("Arial", 20), fg="#e74c3c", bg="#161b22").pack(anchor="center")

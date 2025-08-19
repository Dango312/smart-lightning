import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, request, jsonify
import joblib

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

model = joblib.load('models/gesture_model.joblib')
GESTURE_CLASSES = ['NONE', 'PEACE', 'THUMBS_DOWN', 'THUMBS_UP'] 

app = Flask(__name__)

def normalize_landmarks(landmarks_list):
    points = np.array([[lm.x, lm.y] for lm in landmarks_list])
    wrist = points[0]
    points = points - wrist
    max_dist = np.max(np.linalg.norm(points, axis=1))
    if max_dist > 0:
        points = points / max_dist
    return points.flatten()

@app.route('/recognize', methods=['POST'])
def recognize():
    file = request.files['image']
    filestr = file.read()
    npimg = np.frombuffer(filestr, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    gesture = "NONE"
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        normalized_coords = normalize_landmarks(hand_landmarks.landmark)
        
        prediction = model.predict([normalized_coords])[0]
        gesture = GESTURE_CLASSES[prediction]
        #print(f"GESTURE : {gesture}")

    return jsonify({'gesture': gesture})

if __name__ == '__main__':
    print("--- Hand Gesture Recognition Server is running ---")
    app.run(host='0.0.0.0', port=5001)

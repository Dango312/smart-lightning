import cv2
import mediapipe as mp
import numpy as np
import joblib

class GestureRecognizerPython:
    def __init__(self, model_path, encoder_path):
        print("Python: Initializing MediaPipe Holistic...")
        self.holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5)
        
        print(f"Python: Loading model from {model_path}")
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)
        print("Python: Models loaded successfully.")

    def normalize_landmarks(self, landmarks_row):
        pose_landmarks = np.array(landmarks_row[0:66]).reshape(33, 2)
        lh_landmarks = np.array(landmarks_row[66:108]).reshape(21, 2)
        rh_landmarks = np.array(landmarks_row[108:150]).reshape(21, 2)

        if not np.all(pose_landmarks == 0):
            shoulder_center = (pose_landmarks[11] + pose_landmarks[12]) / 2
            pose_landmarks = pose_landmarks - shoulder_center
            torso_size = np.linalg.norm(shoulder_center - (pose_landmarks[23] + pose_landmarks[24]) / 2)
            if torso_size > 0.01:
                pose_landmarks = pose_landmarks / torso_size

        if not np.all(lh_landmarks == 0):
            wrist = lh_landmarks[0]
            lh_landmarks = lh_landmarks - wrist
            max_dist = np.max(np.linalg.norm(lh_landmarks, axis=1))
            if max_dist > 0:
                lh_landmarks = lh_landmarks / max_dist
                
        if not np.all(rh_landmarks == 0):
            wrist = rh_landmarks[0]
            rh_landmarks = rh_landmarks - wrist
            max_dist = np.max(np.linalg.norm(rh_landmarks, axis=1))
            if max_dist > 0:
                rh_landmarks = rh_landmarks / max_dist

        return np.concatenate([pose_landmarks.flatten(), lh_landmarks.flatten(), rh_landmarks.flatten()])

    def recognize(self, image_bytes: bytes) -> str:
        npimg = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if image is None:
            return "NONE"

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image_rgb)

        gesture = "NONE"
        try:
            if results.pose_landmarks:
                pose = results.pose_landmarks.landmark
                lh = results.left_hand_landmarks.landmark if results.left_hand_landmarks else [type('obj', (object,), {'x': 0, 'y': 0})() for _ in range(21)]
                rh = results.right_hand_landmarks.landmark if results.right_hand_landmarks else [type('obj', (object,), {'x': 0, 'y': 0})() for _ in range(21)]

                pose_row = list(np.array([[lm.x, lm.y] for lm in pose]).flatten())
                lh_row = list(np.array([[lm.x, lm.y] for lm in lh]).flatten())
                rh_row = list(np.array([[lm.x, lm.y] for lm in rh]).flatten())
            
                row = pose_row + lh_row + rh_row
                normalized_row = self.normalize_landmarks(row)
                prediction = self.model.predict([normalized_row])[0]
                gesture = self.label_encoder.inverse_transform([prediction])[0]
        except Exception as e:
            print(f"Python Error: {e}")
            gesture = "NONE"
        
        return gesture


recognizer_instance = GestureRecognizerPython(
    'models/gesture_model.joblib',
    'models/label_encoder.joblib'
)

def recognize_gestures(image_bytes: bytes) -> str:
    return recognizer_instance.recognize(image_bytes)

    
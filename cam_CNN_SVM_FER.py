import cv2
import numpy as np
import joblib
from tensorflow.keras.models import load_model, Model

# 1. Load Saved Models
cnn = load_model("cnn_svm_FER_model.h5")
svm = joblib.load("svm_model.pkl")

# 2. Slice CNN to feature extractor
feature_model = Model(inputs=cnn.input, outputs=cnn.get_layer("feature_layer").output)

# 3. Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 4. Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 5. Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Crop and preprocess face
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48, 48))
        face_normalized = face_resized / 255.0
        face_input = np.expand_dims(face_normalized, axis=(0, -1))  # shape: (1, 48, 48, 1)

        # Predict
        features = feature_model.predict(face_input)
        pred = svm.predict(features)[0]
        emotion = emotion_labels[pred]

        # Draw box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 200, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)
        break  # only 1 face

    cv2.imshow("Emotion Detection (CNN + SVM)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

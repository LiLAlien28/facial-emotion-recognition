import gradio as gr
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("fer_emotion_model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_emotion(image):
    """Detect faces and emotions in the image"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # Process each face
    for (x, y, w, h) in faces:
        # Extract face ROI
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)
        
        # Predict emotion
        prediction = model.predict(roi, verbose=0)
        emotion_idx = np.argmax(prediction)
        emotion = emotion_labels[emotion_idx]
        confidence = float(prediction[0][emotion_idx])
        
        # Draw rectangle and label
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"{emotion} ({confidence:.2%})"
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (0, 255, 0), 2)
    
    return image

# Create Gradio interface
demo = gr.Interface(
    fn=detect_emotion,
    inputs=gr.Image(sources=["webcam", "upload"], type="numpy"),
    outputs=gr.Image(type="numpy"),
    title="🎭 Facial Emotion Recognition",
    description="Upload an image or use your webcam to detect emotions in

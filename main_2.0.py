import tensorflow
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import os
from time import sleep


# Load the face detector and emotion classifier
face_classifier = cv2.CascadeClassifier(r'C:\Users\Bruger\OneDrive\Dokumenter\ec_utbildning\deep_learning\kunskapskontroll\Emotion_Detection_CNN-main\haarcascade_frontalface_default.xml')
classifier = load_model(r'C:\Users\Bruger\OneDrive\Dokumenter\ec_utbildning\deep_learning\kunskapskontroll\Emotion_Detection_CNN-main\model_1.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

camera_index = 0 
cap = cv2.VideoCapture(camera_index)

captured_images = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    if frame is None or frame.size == 0:
        print("Empty frame grabbed")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    # Create a square and print the prediction
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Emotion Detector', frame)
    key = cv2.waitKey(1) & 0xFF
    
    # Possibility close the frame as well as capture and display the image as well as 
    if key == ord('q'):
        print("Exiting...")
        break
    elif key == ord('p'):
        # Capture and save the current frame
        captured_images.append(frame)
        print(f"Captured image {len(captured_images)}")

        cv2.imshow(f'Captured Image {len(captured_images)}', frame)
        cv2.waitKey(500) 

cap.release()
cv2.destroyAllWindows()

# Create a directory to save the images if it doesn't exist and ave the captured images
output_dir = r"C:\Users\Bruger\OneDrive\Dokumenter\ec_utbildning\deep_learning\kunskapskontroll\Emotion_Detection_CNN-main\captured_images"
os.makedirs(output_dir, exist_ok=True)

for i, img in enumerate(captured_images):
    img_path = os.path.join(output_dir, f'captured_image_{i+1}.jpg')
    cv2.imwrite(img_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    print(f"Saved {img_path}")


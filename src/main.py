import face_recognition
import numpy as np
import cv2
from PIL import Image, ImageDraw
from datetime import datetime
import os
import pickle
import pandas as pd
import csv


MODEL_FILE = os.path.join(os.getcwd(), 'src', 'face_encodings.pkl')
ATTENDANCE_FILE = os.path.join(os.getcwd(), 'data', 'attendance.csv')
PHOTOS_DIRECTORY = os.path.join(os.getcwd(), 'data', 'photos')


def encode_faces(directory=PHOTOS_DIRECTORY, model_file=MODEL_FILE):
    """Encode faces and save to a file."""
    known_face_encodings = []
    known_face_names = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            filepath = os.path.join(directory, filename)
            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                face_encoding = encodings[0]
                known_face_encodings.append(face_encoding)
                known_face_names.append(os.path.splitext(filename)[0])
    
    with open(model_file, 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)
    
    print(f"Done encoding faces and saving profiles to {model_file}")
    return known_face_encodings, known_face_names


def load_known_faces(model_file=MODEL_FILE):
    """Load known face encodings and names from a file."""
    if os.path.exists(model_file):
        with open(model_file, 'rb') as f:
            known_face_encodings, known_face_names = pickle.load(f)
        print(f"Loaded face encodings from {model_file}")
    else:
        known_face_encodings, known_face_names = encode_faces()
    return known_face_encodings, known_face_names


def make_attendance_entry(name):
    """Mark attendance for recognized faces using CSV files."""
    if name != "Unknown":
        now = datetime.now()
        date_string = now.strftime('%d/%b/%Y')
        time_string = now.strftime('%H:%M:%S')
        
        if not os.path.exists(ATTENDANCE_FILE):
            with open(ATTENDANCE_FILE, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Name', 'Date', 'Time'])
                return
        
        df = pd.read_csv(ATTENDANCE_FILE)
        
        if not ((df['Name'] == name) & (df['Date'] == date_string)).any():
            with open(ATTENDANCE_FILE, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([name, date_string, time_string])
                print(f"Attendance recorded for {name} on {date_string} at {time_string}")


def initialize_webcam():
    """Initialize webcam capture with error handling."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    return cap


def process_frame(frame, known_face_encodings, known_face_names):
    """Process a single frame for face recognition and mark attendance."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    pil_image = Image.fromarray(rgb_frame)
    draw = ImageDraw.Draw(pil_image)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 255))
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        make_attendance_entry(name)

    return frame


def main():
    known_face_encodings, known_face_names = load_known_faces()
    cap = initialize_webcam()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image from webcam.")
            break

        frame = process_frame(frame, known_face_encodings, known_face_names)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

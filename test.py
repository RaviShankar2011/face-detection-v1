import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
import os

# Load known faces dynamically from the "images" directory
known_faces = {}
    name, _ = os.path.splitext(file_name)
for file_name in os.listdir("images"):
    image_path = os.path.join("images", file_name)
    user_image = face_recognition.load_image_file(image_path)
    user_face_encoding = face_recognition.face_encodings(user_image)[0]
    known_faces[name] = user_face_encoding

# List of students expected
students = list(known_faces.keys())

# Open CSV file for writing attendance
now = datetime.now()
current_date = now.strftime("%d-%m-%y")
csv_file_path = f"{current_date}.csv"
csv_file = open(csv_file_path, "w+", newline="")
csv_writer = csv.writer(csv_file)

# Open video capture
video_capture = cv2.VideoCapture(0)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        # Compare with known faces
        matches = face_recognition.compare_faces(list(known_faces.values()), face_encoding)
        face_distances = face_recognition.face_distance(list(known_faces.values()), face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = list(known_faces.keys())[best_match_index]

            # Add text to image
            if name in students:
                students.remove(name)
                current_time = now.strftime("%H-%M-%S")
                csv_writer.writerow([name, current_time])

            font = cv2.FONT_HERSHEY_SIMPLEX
            bottom_left_corner_of_text = (10, 100)
            font_scale = 1.5
            font_color = (255, 0, 0)
            thickness = 3
            line_type = 2
            cv2.putText(frame, f"{name} ", bottom_left_corner_of_text, font, font_scale, font_color, thickness, line_type)

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
csv_file.close()

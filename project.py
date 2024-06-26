
import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime


# for reading from webcam
video_capture=cv2.VideoCapture(0);

#load sample pic for ravi
ravi1_image=face_recognition.load_image_file("images/ravi.jpg")
ravi1_face_encoding=face_recognition.face_encodings(ravi1_image)[0]

#load sample pic for aman
aman1_image=face_recognition.load_image_file("images/aman.jpg")
aman1_face_encoding=face_recognition.face_encodings(aman1_image)[0]

#load sample pic for aman
aman2_image=face_recognition.load_image_file("images/aman.jpg")
aman2_face_encoding=face_recognition.face_encodings(aman2_image)[0]

#load sample pic for ravi
ravi2_image=face_recognition.load_image_file("images/ravi.jpg")
ravi2_face_encoding=face_recognition.face_encodings(ravi2_image)[0]

#array of known faces and thier names
known_face_encodings = [
    ravi1_face_encoding,
    aman1_face_encoding,
    ravi2_face_encoding,
    aman2_face_encoding
]
known_face_names = [
    "ravi1",
    "aman1",
    "ravi2",
    "samman"
]

#list of students expected
students=known_face_names.copy()

face_locations = []
face_encodings = []

#get current date and time
now = datetime.now()
current_date = now.strftime("%d-%m-%y")

f = open(f"{current_date}.csv","w+",newline="")
lnwriter = csv.writer(f)

while True :
    _, frame =  video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame=cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)

    #recognize faces
    face_locations=face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings,face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings,face_encoding)
        best_match_index = np.argmin(face_distance)

        if(matches[best_match_index]):
            name= known_face_names[best_match_index]

        #add text to image
        if name in known_face_names:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOftext= (10,100)
            fontScale =1.5
            fontColor=(255,0,0)
            thickness =3
            lineType =2
            cv2.putText(frame,name + " present",bottomLeftCornerOftext,font,fontScale,fontColor,thickness,lineType)

            if name in students:
                students.remove(name)
                current_time=now.strftime("%H-%M-%S")
                lnwriter.writerow([name,current_time])


    cv2.imshow("Attendence",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()

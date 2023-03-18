# edited files:
#     - /Volumes/Shandilya/SGP/Environments/medpipe/lib/python3.7/site-packages/torchvision/models/maxvit.py
#     - /Volumes/Shandilya/SGP/Environments/medpipe/lib/python3.7/site-packages/tensorflow/core/function/polymorphism/function_type.py

# imports
import cv2
import json
import numpy as np
from face_detection import RetinaFace

preferences_file = "preferences.json"

f = open(preferences_file, 'r')
preferences = json.load(f)

# retina face
detector = RetinaFace()

def recognize_face(frame):
    width, height = preferences["face"]["width"], preferences["face"]["height"]
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    faces = detector(frame)

    if len(faces) == 0:
        return {"success": False, "detections": []}

    persons = []

    for face in faces:
        print(face[2])
        if face[2] > preferences["face"]["min_detection_confidence"]:
            # persons.append({"bounding_box": face[0], "id": None})
            bounding_box = (face[0][0]/width, face[0][1]/height, face[0][2]/width, face[0][3]/height)
            persons.append({"bounding_box": bounding_box, "id": None})

    return {"success": True, "detections": persons}

if __name__ == "__main__":
    capture = cv2.VideoCapture(0)
    while capture.isOpened():
        ret, frame = capture.read()
        height, width, depth = frame.shape

        if not ret:
            print("frame empty!")

        result = recognize_face(frame)

        if result["success"]:
            for person in result["detections"]:
                # bounding_box = [int(x) for x in person["bounding_box"]]
                # frame = cv2.rectangle(frame, tuple(bounding_box[:2]), tuple(bounding_box[2:]), (255, 0, 0), 2)

                bounding_box = person["bounding_box"]
                print(bounding_box)
                frame = cv2.rectangle(frame, (int(bounding_box[0] * width), int(bounding_box[1] * height)), (int(bounding_box[2] * width), int(bounding_box[3] * height)), (255, 0, 0), 2)

        cv2.imshow('Face Detection and Recognition Demo', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


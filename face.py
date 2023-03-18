# edited files:
#     - /Volumes/Shandilya/SGP/Environments/medpipe/lib/python3.7/site-packages/torchvision/models/maxvit.py
#     - /Volumes/Shandilya/SGP/Environments/medpipe/lib/python3.7/site-packages/tensorflow/core/function/polymorphism/function_type.py

# imports
import cv2
import json
import numpy as np
import mediapipe as mp

from face_detection import RetinaFace

preferences_file = "preferences.json"

f = open(preferences_file, 'r')
preferences = json.load(f)

# mediapipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# retina face
detector = RetinaFace()

def medpipe_recognize(frame):
    height, width, depth = frame.shape
    # with mp_face_detection.FaceDetection(model_selection = preferences["face"]["model_selection"], 
    #         min_detection_confidence = preferences["face"]["min_detection_confidence"]) as face_detection:
    with mp_face_detection.FaceDetection(min_detection_confidence = preferences["face"]["min_detection_confidence"]) as face_detection:
        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.detections:
            return {"success": False, "detections": []}

        persons = []
        for detection in results.detections:
            relative_data = detection.location_data.relative_bounding_box
            xmin, ymin, w, h = relative_data.xmin, relative_data.ymin, relative_data.width, relative_data.height
            bounding_box = (xmin * width, ymin * height, w * width, h * height)

            persons.append({"bounding_box": bounding_box, "id": None})

    return {"success": True, "detections": persons}

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

capture = cv2.VideoCapture(0)

if __name__ == "__main__":
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


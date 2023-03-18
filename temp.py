import cv2
from face_detection import RetinaFace

detector = RetinaFace()
img = cv2.imread("crowds.jpg")
faces = detector(img)
box, landmarks, score = faces[0]

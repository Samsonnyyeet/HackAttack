import cv2
import numpy as np
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image

# Load the RetinaNet model
model_path = 'data/resnet50_coco_best_v2.1.0.h5'
model = models.load_model(model_path, backbone_name='resnet50')

# Initialize the video stream
video_path = 'data/sample_video.mp4'
cap = cv2.VideoCapture(video_path)

# Loop over the frames in the video stream
while True:
    # Read the next frame from the video stream
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    image = preprocess_image(frame)
    image, scale = resize_image(image)

    # Perform object detection
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

    # Rescale the boxes
    boxes /= scale

    # Loop over the detections and draw boxes around the objects
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < 0.5:
            continue
        box = box.astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), thickness=2)
        cv2.putText(frame, str(label) + ': ' + str(score), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)

    # Show the frame
    cv2.imshow('frame', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()


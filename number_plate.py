import cv2
import numpy as np

# Load YOLOv4 model
# net = cv2.dnn.readNet('C:\Users\Sai Tarun\OneDrive\Desktop\python\yolov4.weights', 'C:\Users\Sai Tarun\OneDrive\Desktop\python\yolov4.cfg')

net = cv2.dnn.readNet("yolo/yolov4.weights", "yolo/yolov4.cfg")

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        # Read a frame
        ret, frame = cap.read()

        # Resize the frame to the input size required by YOLOv4
        resized = cv2.resize(frame, (100, 100))

        # Create a blob from the resized image
        blob = cv2.dnn.blobFromImage(resized, 1/255.0, (608, 608), swapRB=True, crop=False)

        # Pass the blob through the YOLOv4 model
        net.setInput(blob)
        outputs = net.forward(net.getUnconnectedOutLayersNames())

        # Filter out car detections
        cars = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                if class_id == 2:  # Car class
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * frame.shape[1])
                        center_y = int(detection[1] * frame.shape[0])
                        w = int(detection[2] * frame.shape[1])
                        h = int(detection[3] * frame.shape[0])
                        x = center_x - w // 2
                        y = center_y - h // 2
                        cars.append((x, y, w, h))

        # Draw bounding boxes around car detections
        for x, y, w, h in cars:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the result
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

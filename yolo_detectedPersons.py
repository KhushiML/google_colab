import cv2
import numpy as np
def detect_faces_yolo(image):
    # Load YOLO model
    yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = yolo_net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in yolo_net.getUnconnectedOutLayers()]

    # Prepare the image for YOLO
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    outputs = yolo_net.forward(output_layers)

    # Collect bounding boxes for detected persons
    boxes = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]  # Get the scores for each class
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == 0 and confidence > 0.5:  # Class ID 0 is 'person'
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append((x, y, w, h))

    face_eye_data = []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Loop over detected person boxes and detect faces within
    for (x, y, w, h) in boxes:
        roi = image[y:y+h, x:x+w]  # Region of Interest (person area)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_roi, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) >= 1:
            face_eye_data.append(((x, y, w, h), faces))

    return face_eye_data

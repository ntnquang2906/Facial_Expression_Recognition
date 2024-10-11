import cv2
from keras.models import model_from_json
import numpy as np
import time

# Load your model and cascade classifier
json_file = open(r"C:\Users\quang\OneDrive\Máy tính\src\best_model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights(r"C:\Users\quang\OneDrive\Máy tính\src\best_model.hdf5")
haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_file)

# Define a function to extract features from an image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Initialize the webcam
webcam = cv2.VideoCapture(0)
labels = {0: "normal", 1: "small_smile", 2: "smile", 3: "big_smile"}

# Initialize frame count and time variables
frame_count = 0
start_time = time.time()

while True:
    i, im = webcam.read()

    frame_count += 1

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(im, 1.3, 5)

    try:
        for p, q, r, s in faces:
            image = gray[q:q + s, p:p + r]
            cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            cv2.putText(im, "%s" % (prediction_label), (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

        cv2.imshow("Output", im)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    except cv2.error:
        pass

    # Check if 1 second has elapsed
    end_time = time.time()
    elapsed_time = end_time - start_time
    if elapsed_time >= 1.0:
        print(f"Frames processed in 1 second: {frame_count}")
        frame_count = 0
        start_time = time.time()

# Release the webcam and close all windows
webcam.release()
cv2.destroyAllWindows()

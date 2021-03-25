# USAGE
# python liveness_demo.py --model model --le le.pickle --detector face_detector

# import the necessary packages
from imutils.video import VideoStream
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils import face_utils
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import dlib

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to trained model")
ap.add_argument("-l", "--le", type=str, required=True,
                help="path to label encoder")
ap.add_argument("-d", "--detector", type=str, required=True,
                help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
                              "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load the liveness detector model and label encoder from disk
print("[INFO] loading liveness detector...")
model = load_model(args["model"])
le = pickle.loads(open(args["le"], "rb").read())

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Facial landmarks
facial_landmarks = {
    "mouth": list(range(48, 68)),
    "right_eyebrow": list(range(17, 22 + 1)),
    "left_eyebrow": list(range(22, 27 + 1)),
    "right_eye": list(range(36, 42 + 1)),
    "left_eye": list(range(42, 48 + 1)),
    "nose": list(range(27, 35 + 1)),
    "jaw": list(range(0, 17 + 1))
}

# initialize module to detect facial landmark
predictor_path = './facial_landmark/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

BOUNDING_BOX_START_POINT = (150, 100)
BOUNDING_BOX_END_POINT = (400, 400)


def mouth_rate(mouth):
    # compute the euclidean distances between the two sets of
    # vertical mouth landmarks (x, y)-coordinates
    A = dist.euclidean(mouth[2], mouth[10])  # 51, 59
    B = dist.euclidean(mouth[4], mouth[8])  # 53, 57

    # compute the euclidean distance between the horizontal
    # mouth landmark (x, y)-coordinates
    C = dist.euclidean(mouth[0], mouth[6])  # 49, 55

    # compute the mouth aspect ratio
    mar = (A + B) / (2.0 * C)

    # return the mouth aspect ratio
    return mar


def eye_aspect_ratio(eye):
    # Detect does human open eyes
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    # compute the eye_aspect_ratio
    aer = (A + B) / (2.0 * C)

    # return eye_aspect_ratio
    return aer


def point_in_box(point, bounding_box_point):
    return True


# loop over the frames from the video stream
while True:
    # Flag to detect
    eye_open_flag = False
    Face_in_box_flag = False
    Open_mouth_flag = False

    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 600 pixels
    frame = vs.read()
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=600)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # DETECT FACE
    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # Bounding box for face in
    cv2.rectangle(frame, BOUNDING_BOX_START_POINT, BOUNDING_BOX_END_POINT, (255, 0, 0), 10)

    # loop over the detections
    for j in range(0, detections.shape[2]):

        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, j, 2]

        # filter out weak detections
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the face and extract the face ROI
            box = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # print("startX, startY, endX, endY:{}, {}, {}, {}".format(startX, startY, endX, endY))

            # If face in bounding box
            if ((startX >= BOUNDING_BOX_START_POINT[0] and startY >= BOUNDING_BOX_START_POINT[1])
                    and
                    (endX <= BOUNDING_BOX_END_POINT[0] and endY <= BOUNDING_BOX_END_POINT[1])):

                # Change color of bounding box
                cv2.rectangle(frame, BOUNDING_BOX_START_POINT, BOUNDING_BOX_END_POINT, (0, 255, 0), 10)

                # ensure the detected bounding box does fall outside the
                # dimensions of the frame
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                # extract the face ROI and then preproces it in the exact
                # same manner as our training data
                face = frame[startY:endY, startX:endX]
                face = face.astype("float") / 255.0
                try:
                    face = cv2.resize(face, (32, 32))
                except:
                    print("Hell")

                face = img_to_array(face)
                face = np.expand_dims(face, axis=0)

                # pass the face ROI through the trained liveness detector
                # model to determine if the face is "real" or "fake"
                preds = model.predict(face)[0]
                j = np.argmax(preds)
                label = le.classes_[j]

                print("Label: {}".format(label))

                # draw the label and bounding box on the frame
                label = "{}: {:.4f}".format(label, preds[j])
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)

                ## # loop over the face detections
                for (i, rect) in enumerate(rects):
                    # Facial landmarks
                    # determine the facial landmarks for the face region, then
                    # convert the facial landmark (x, y)-coordinates to a NumPy
                    # array
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)

                    copy_img = frame.copy()

                    # Drawing facial landmark
                    for (x, y) in shape:
                        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

                    # mouth by mouth
                    list_mouth_points = shape[facial_landmarks['mouth']]
                    list_mouth_points = shape[facial_landmarks['mouth']]

                    mouth_AER = mouth_rate(list_mouth_points)
                    print("MOUNT RATE: {}".format(mouth_AER))

                    # Eyes
                    list_right_eye_points = shape[facial_landmarks['right_eye']]
                    list_left_eye_points = shape[facial_landmarks['left_eye']]

                    right_eye_AER = eye_aspect_ratio(list_right_eye_points)
                    left_eye_AER = eye_aspect_ratio(list_left_eye_points)

                    # print("left_eye_points, right_eye_points: {}".format(left_eye_AER, right_eye_AER))

                    # Check eyes are open
                    if (right_eye_AER > 0.1 and left_eye_AER > 0.1):
                        eye_open_flag = True

                    # Check Mouth open
                    if (mouth_AER > 0.8):
                        Open_mouth_flag = True

                    # If mouth is open then verify video
                    if (eye_open_flag and Open_mouth_flag):

                        # Confirmed
                        if "real" in label:
                            label = "{}".format("Verified")
                            cv2.putText(frame, label, (startX, endY - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Rejected
                        if "fake" in label:
                            label = "{} ".format("Rejected")
                            cv2.putText(frame, label, (startX, endY - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # show the output frame and wait for a key press
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

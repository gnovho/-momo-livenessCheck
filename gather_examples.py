# USAGE
# python gather_examples.py --input videos/real.mov --output dataset/real --detector face_detector --skip 1
# python gather_examples.py --input videos/fake.mov --output dataset/fake --detector face_detector --skip 4

# import the necessary packages
import numpy as np
import argparse
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
                help="path to input video")
ap.add_argument("-o", "--output", type=str, required=True,
                help="path to output directory of cropped faces")
ap.add_argument("-d", "--detector", type=str, required=True,
                help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip", type=int, default=16,
                help="# of frames to skip before applying face detection")
ap.add_argument("-r", "--ratio", type=float, default=0.04,
                help="# ratio expand face crop")
args = vars(ap.parse_args())


def expand_facebox(startX, startY, endX, endY, w, h, ratio=0.3):
    X_start = startX - ratio * startX
    Y_start = startY - ratio * startY
    X_end = endX + ratio * endX
    Y_end = endY + ratio * endY

    startX = max(0, X_start)
    startY = max(0, Y_start)
    endX = min(w, X_end)
    endY = min(h, Y_end)

    startX_ = max(0, startX)
    startY_ = max(0, startY)
    endX_ = min(w, endX)
    endY_ = min(h, endY)
    return int(startX_), int(startY_), int(endX_), int(endY_)


# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
                              "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# open a pointer to the video file stream and initialize the total
# number of frames read and saved thus far
vs = cv2.VideoCapture(args["input"])
read = 0
saved = 0

# loop over frames from the video file stream
while True:
    # grab the frame from the file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # increment the total number of frames read thus far
    read += 1

    # check to see if we should process this frame
    if read % args["skip"] != 0:
        continue

    # grab the frame dimensions and construct a blob from the frame
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # ensure at least one face was found
    if len(detections) > 0:
        # we're making the assumption that each image has only ONE
        # face, so find the bounding box with the largest probability
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        # ensure that the detection with the largest probability also
        # means our minimum probability test (thus helping filter out
        # weak detections)
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the face and extract the face ROI
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            startX_ext, startY_ext, endX_ext, endY_ext = expand_facebox(
                startX, startY, endX, endY,
                w, h, args["ratio"])

            face = frame[startY_ext:endY_ext, startX_ext:endX_ext]

            # write the frame to disk
            p = os.path.sep.join([args["output"],
                                  "{}.png".format(saved)])
            cv2.imwrite(p, face)
            saved += 1
            print("[INFO] saved {} to disk".format(p))

# do a bit of cleanup
vs.release()
cv2.destroyAllWindows()

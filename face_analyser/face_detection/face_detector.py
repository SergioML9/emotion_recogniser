import cv2
import numpy as np
import configuration.face_detection_settings as face_detection_settings

def changeColorScale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def equalizeHistogram(frame):
    return cv2.equalizeHist(frame)

def getFaceCoordinates(frame):

    cascade = cv2.CascadeClassifier(face_detection_settings.cascade_path)

    img_equalized = equalizeHistogram(frame)

    rects = cascade.detectMultiScale(img_equalized,
        scaleFactor=face_detection_settings.scale_factor,
        minNeighbors=face_detection_settings.min_neighbors,
        minSize=face_detection_settings.min_size
    )

    # For now, we only deal with the case that we detect one face.
    if(len(rects) <= 0) :
        return None

    # get first face
    face = rects[0]
    bounding_box = [face[0], face[1], face[0] + face[2], face[1] + face[3]]

    return bounding_box

def cropFace(frame, face_coordinates):
    return frame[face_coordinates[1]:face_coordinates[3], face_coordinates[0]:face_coordinates[2]]

def preprocess(frame, face_coordinates):
    face = cropFace(frame, face_coordinates)
    face_resized = cv2.resize(face, face_detection_settings.face_shape)
    input_img = np.expand_dims(face_resized, axis=0)
    return np.expand_dims(input_img, axis=0)

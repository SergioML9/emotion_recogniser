import cv2
import numpy as np
import configuration.video_receiver_settings as video_receiver_settings

def initializeVideoCapture():
    return cv2.VideoCapture(0)

def captureFrame(video_capture):
    return video_capture.read()

def displayFrame(frame):
    if video_receiver_settings.show_cam:
        cv2.imshow(video_receiver_settings.webcam_window_name, frame)

def checkVideo():
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    return True

def stopVideoCapture():
    cv2.destroyAllWindows()

def drawFace(frame, face_coordinates):
    if video_receiver_settings.show_cam:
        cv2.rectangle(np.asarray(frame), (face_coordinates[0], face_coordinates[1]),
            (face_coordinates[2], face_coordinates[3]),
            video_receiver_settings.face_color,
            thickness=video_receiver_settings.thickness)
        displayFrame(frame)

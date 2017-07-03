import cv2
import numpy as np
import configuration.video_receiver_settings as video_receiver_settings

def initializeVideoCapture():
    video_capture = cv2.VideoCapture(0)
    video_capture.set(3,600)
    video_capture.set(4,400)
    return video_capture

def captureFrame(video_capture):
    return video_capture.read()

def displayFrame(frame):
    if video_receiver_settings.show_cam:
        cv2.imshow(video_receiver_settings.webcam_window_name, frame)
        cv2.imshow(video_receiver_settings.emoji_window_name, frame)

def displayEmojiFrame(frame):
    if video_receiver_settings.show_cam:
        cv2.imshow(video_receiver_settings.emoji_window_name, frame)

def checkVideo():
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    return True

def stopVideoCapture():
    cv2.destroyAllWindows()

def drawFace(frame, face_coordinates, emotion):
    if video_receiver_settings.show_cam:
        emoji = cv2.imread('emoji/' + emotion + ".png")


        displayFrame(frame)
        
        x = face_coordinates[0]
        y = face_coordinates[1]
        h = face_coordinates[2]
        w = face_coordinates[3]

        for c in range(0,3):
            frame[y:y+emoji.shape[0], x:x+emoji.shape[1], c] = emoji[:,:,c] * (emoji[:,:,2]/255.0) +  frame[y:y+emoji.shape[0], x:x+emoji.shape[1], c] * (1.0 - emoji[:,:,2]/255.0)

        #frame[y:y+emoji.shape[0], x:x+emoji.shape[1]] = emoji

        #print(face_coordinates)
        #frame[(face_coordinates[0], face_coordinates[1]), (face_coordinates[2], face_coordinates[3])] = emoji
        #cv2.rectangle(np.asarray(frame), (face_coordinates[0], face_coordinates[1]),
        #    (face_coordinates[2], face_coordinates[3]),
        #    video_receiver_settings.face_color,
        #    thickness=video_receiver_settings.thickness)
        displayEmojiFrame(frame)

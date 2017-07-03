import face_detection.video_receiver as video_receiver
import face_detection.face_detector as face_detector
import configuration.general_settings as settings

from model.vgg_adapted_model import FaceAnalyserModel

def main():

    # Initialize model
    model = FaceAnalyserModel(settings.model_weights_path)

    # Initialize video capture
    capture = video_receiver.initializeVideoCapture()

    # Check is user wants quit
    while(video_receiver.checkVideo()):

        # Capture frame-by-frame
        ret, frame = video_receiver.captureFrame(capture)

        # Display the resulting frame
        video_receiver.displayFrame(frame)

        # Change image scale color
        frame_gray = face_detector.changeColorScale(frame)

        # Detect face
        face_coordinates = face_detector.getFaceCoordinates(frame_gray)

        # Check if face has been detected
        if face_coordinates is not None:

            # Preprocess image
            preprocessed_frame = face_detector.preprocess(frame_gray, face_coordinates)

            emotion_prob, emotion_index = model.detectEmotion(preprocessed_frame)
            #print ("The worker is " + settings.detected_emotions[emotion_index])
            print("ex:EmotionDetected rdf:type ewe-emodet:EmotionDetected. \
                    \n ex:EmotionDetected ewe-emodet:hasDetected onyx:Emotion. \
                    \n onyx:Emotion onyx:hasEmotionCategory wn-affect:" + settings.detected_emotions[emotion_index] + " . \
                    \n onyx:Emotion onyx:hasEmotionIntensity " + str(emotion_prob) + ".\n\n")
            # Draw face rectangle
            video_receiver.drawFace(frame, face_coordinates, settings.detected_emotions[emotion_index])

    # When everything done, release the capture
    capture.release()
    video_receiver.stopVideoCapture()

if __name__ == '__main__':
    main()

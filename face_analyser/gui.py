from tkinter import *

import csv
import face_detection.video_receiver as video_receiver
import face_detection.face_detector as face_detector
import configuration.general_settings as settings

import data.data_loader as data_loader

from model.vgg_adapted_model import FaceAnalyserModel

class App:

    def __init__(self, master):
        frame = Frame(master)
        frame.pack()

        self.btnDetectEmotion = Button(frame,
                         text="Start emotion detection", fg="red",
                         command=self.startEmotionDetection)
        self.btnDetectEmotion.pack(side=LEFT)

        self.btnTrainModel = Button(frame,
                         text="Train Model", fg="red",
                         command=self.trainModel)
        self.btnTrainModel.pack(side=LEFT)

        self.btnEvaluateModel = Button(frame,
                         text="Evaluate Model", fg="red",
                         command=self.evaluateModel)
        self.btnEvaluateModel.pack(side=LEFT)

        self.btnGetData = Button(frame,
                         text="Get data from dataset", fg="red",
                         command=self.generateDataFromDataset)
        self.btnGetData.pack(side=LEFT)

        self.btnQuit = Button(frame,
                         text="Exit",
                         command=quit)
        self.btnQuit.pack(side=LEFT)

    def startEmotionDetection(self):
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
                # Draw face rectangle
                video_receiver.drawFace(frame, face_coordinates)

                # Preprocess image
                preprocessed_frame = face_detector.preprocess(frame_gray, face_coordinates)

                emotion_index = model.detectEmotion(preprocessed_frame)
                print (settings.detected_emotions[emotion_index])

        # When everything done, release the capture
        capture.release()
        video_receiver.stopVideoCapture()

    # def generateDataFromDataset(self):
    #     print('Obtaining data from dataset')
    #     csvr = csv.reader(open('data/fer2013.csv'))
    #     header = next(csvr)
    #     rows = [row for row in csvr]
    #
    #     trn = [row[:-1] for row in rows if row[-1] == 'Training']
    #     csv.writer(open('data/training.csv', 'w+')).writerows([header[:-1]] + trn)
    #     print("Training: " + str(len(trn)))
    #
    #     tst = [row[:-1] for row in rows if row[-1] == 'PublicTest']
    #     csv.writer(open('data/test.csv', 'w+')).writerows([header[:-1]] + tst)
    #     print("PublicTest: " + str(len(tst)))
    #
    #     tst2 = [row[:-1] for row in rows if row[-1] == 'PrivateTest']
    #     csv.writer(open('data/testprivate.csv', 'w+')).writerows([header[:-1]] + tst2)
    #     print("PrivateTest: " + str(len(tst2)))

    def generateDataFromDataset(self):
        data_loader.generate_data()

    def trainModel(self):
        model = FaceAnalyserModel(settings.model_weights_path)
        model.trainModel()

    def evaluateModel(self):
        model = FaceAnalyserModel(settings.model_weights_path)
        model.evaluateModel()

root = Tk()
app = App(root)
root.mainloop()

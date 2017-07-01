import speech_recognition as sr
import time
import emotion_extraction.emotion_extractor as emotion_extractor

# this is called from the background thread
def callback(recognizer, audio):
    # received audio data, now we'll recognize it using Google Speech Recognition
    try:
        #print("Google Speech Recognition thinks you said " + recognizer.recognize_google(audio))
        print("Emotion detected: " + emotion_extractor.detectEmotion(recognizer.recognize_google(audio)))
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

def initializeAudioRecording():

    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        # calibrate noise
        recognizer.adjust_for_ambient_noise(source)

    # start listening in the background
    stop_listening = recognizer.listen_in_background(microphone, callback)

    while True:
        time.sleep(0.1)

# Emotion recognition from face and speech with Python

## Dataset

[fer2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) is the dataset used for training the model.

## How to install

### Create and activate a virtual environment
It is recommended to work in a virtual environment, that can be created and activated with [Anaconda](https://www.continuum.io/downloads) using the following commands:

```
conda create -n emotion_detection python=3.4
source activate emotion_detection
```

### Install dependencies

```
conda install scikit-learn
conda install -c menpo opencv3=3.1.0
pip install --upgrade keras
conda install pandas
conda install h5py
pip install SpeechRecognition
apt-get install portaudio19-dev
pip install pyaudio
```

### Configure Keras

Create the file ```~/.keras/keras.json``` with the following content:

```
{
    "image_dim_ordering": "th",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "theano"
}
```

### Clone the repository

```
git clone https://github.com/SergioML9/emotion_recogniser
```

## Usage

### Emotion detection from face analysis

#### Bash option

Run `face_analyser/run.py` script, and the emotion detection process will start, showing the output in the terminal. Press `q`
in order to stop the detection.

#### GUI option

Run `face_analyser/gui.py` script, and a very simple GUI will be shown. The GUI has five buttons:

1. Start emotion detection: starts the emotion recognition, printing the outputs in the terminal.
2. Train model: starts the model training with the data specified at `configuration/data_settings.py`. 
3. Evaluate model: evaluates the model with the data specified at `configuration/data_settings.py`. 
4. Get data from dataset: converts the data from the csv dataset to npy files.
5. Exit: closes the application.

### Emotion detection from speech analysis

# Bash option

Run `speech_analyser/run.py` script, and the emotion detection process will start, showing the output in the terminal.

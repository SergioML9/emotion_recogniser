from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras import backend as K

import configuration.data_settings as data_settings

import cv2, numpy as np

K.set_image_dim_ordering('th')

class FaceAnalyserModel():

    def __init__(self, weights_path=None, shape=(48, 48)):

        self.model = Sequential()
        self.model.add(ZeroPadding2D((1,1), input_shape=(1, 48, 48)))
        self.model.add(Conv2D(32, (3, 3), activation="relu"))
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Conv2D(32, (3, 3), activation="relu"))
        self.model.add(MaxPooling2D((2,2), strides=(2,2)))

        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Conv2D(64, (3, 3), activation="relu"))
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Conv2D(64, (3, 3), activation="relu"))
        self.model.add(MaxPooling2D((2,2), strides=(2,2)))

        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Conv2D(128, (3, 3), activation="relu"))
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Conv2D(128, (3, 3), activation="relu"))
        self.model.add(ZeroPadding2D((1,1)))
        self.model.add(Conv2D(128, (3, 3), activation="relu"))
        self.model.add(MaxPooling2D((2,2), strides=(2,2)))

        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(6, activation='softmax'))

        print ("Model successfully created")

        if weights_path:
            self.model.load_weights(weights_path)

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def detectEmotion(self, frame):
        result = self.model.predict(frame)[0]
        return np.argmax(result)

    def trainModel(self):

        X_fname = 'data/npy/X_' + data_settings.training_data + '.npy'
        y_fname = 'data/npy/y_' + data_settings.training_data + '.npy'
        X_train = np.load(X_fname)
        y_train = np.load(y_fname)
        print("x shape: " + str(X_train.shape))
        print("y shape: " + str(y_train.shape))

        print("Training started")

        callbacks = []
        earlystop_callback = EarlyStopping(monitor='val_loss', patience=5, verbose=0)
        batch_print_callback = LambdaCallback(on_batch_begin=lambda batch, logs: print(batch))
        epoch_print_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: print("epoch:", epoch))
        callbacks.append(earlystop_callback)
        callbacks.append(batch_print_callback)
        callbacks.append(epoch_print_callback)

        batch_size = 512
        self.model.fit(X_train, y_train, nb_epoch=400, \
                batch_size=batch_size, \
                validation_split=0.2, \
                shuffle=True, verbose=0, \
                callbacks=callbacks)

        self.model.save_weights(data_name + '_weights.h5')
        scores = model.evaluate(X_train, y_train, verbose=0)

        print ("Train loss : %.3f" % scores[0])
        print ("Train accuracy : %.3f" % scores[1])
        print ("Training finished")

    def evaluateModel(self):
        X_fname = 'data/npy/X_' + data_settings.evaluate_data + '.npy'
        y_fname = 'data/npy/y_' + data_settings.evaluate_data + '.npy'
        X_train = np.load(X_fname)
        y_train = np.load(y_fname)
        print("x shape: " + str(X_train.shape))
        print("y shape: " + str(y_train.shape))

        print("Evaluation started")

        scores = self.model.evaluate(X_train, y_train, verbose=0)

        print ("Test loss : %.3f" % scores[0])
        print ("Test accuracy : %.3f" % scores[1])
        print ("Test finished")

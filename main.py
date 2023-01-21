import math
import os.path
import re
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import h5py
import cv2.cv2 as cv2

import tensorflow as tf
from tensorflow.python.keras.utils import to_categorical

font_categories = {
    'Alex Brush': 0,
    'Open Sans': 1,
    'Sansation': 2,
    'Ubuntu Mono': 3,
    'Titillium Web': 4,
}

def preprocess_data(data_folder: str):
    training_file = 'SynthText_train.h5'
    db = h5py.File(os.path.join(data_folder, training_file), 'r')
    dsets = sorted(db['data'].keys())

    data = []
    labels = []

    for k in dsets:
        rgb = db['data'][k][...]
        charsBB = db['data'][k].attrs['charBB']
        wordsBB = db['data'][k].attrs['wordBB']
        txt = db['data'][k].attrs['txt']
        fonts = db['data'][k].attrs['font']

        for font in fonts:
            labels.append(font_categories[font.decode('UTF-8')])

        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        n, m = gray.shape

        txt = [re.split(' \n|\n |\n| ', t.decode('UTF-8').strip()) for t in txt]
        txt = list(chain(*txt))
        txt = [t for t in txt if len(t) > 0]

        for word_index in range(len(txt)):
            for char_index in range(len(txt[word_index])):

                charBB = charsBB[:, :, char_index]
                pts1 = np.float32([[charBB[0][0], charBB[1][0]],
                                   [charBB[0][3], charBB[1][3]],
                                   [charBB[0][1], charBB[1][1]],
                                   [charBB[0][2], charBB[1][2]]])
                height = math.sqrt((charBB[0][0] - charBB[0][3]) ** 2 + (charBB[1][0] - charBB[1][3]) ** 2)
                width = math.sqrt((charBB[0][0] - charBB[0][1]) ** 2 + (charBB[1][0] - charBB[1][1]) ** 2)
                pts2 = np.float32([[0, 0],
                                   [0, height],
                                   [width, 0],
                                   [width, height]])

                M = cv2.getPerspectiveTransform(pts1, pts2)
                cropped_image = cv2.warpPerspective(gray, M, (int(width), int(height)))
                cropped_image = cv2.resize(cropped_image, (28, 28), interpolation = cv2.INTER_AREA)
                data.append(cropped_image)

                # cv2.imshow("cropped", cropped_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

    db.close()

    data = np.asarray(data).reshape((len(data), 28, 28, 1))
    labels = np.asarray(labels)
    labels = to_categorical(labels, 5, dtype='float32')

    return data, labels

def train(data, labels):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 28x28 with 3 bytes color
        # This is the first convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
    #     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    #     tf.keras.layers.MaxPooling2D(2,2),
        # The third convolution
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # The fourth convolution
    #     tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    #     tf.keras.layers.MaxPooling2D(2,2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    # model.summary()
    model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    csv_logger = tf.keras.callbacks.CSVLogger('training.csv', append=True)

    # train model
    model.fit(x_train, y_train,
              epochs=40, batch_size=64,
              validation_data=[x_test, y_test],
              callbacks=[csv_logger])

    model.save("font_basic.h5")

def calculate_accuracy(training_file: str):
    data = np.genfromtxt(training_file, delimiter=',')
    data = data[1:][:, 1:]

    fig, axes = plt.subplots(1, 2)

    # plot train and test accuracies
    axes[0].plot(data[:, 0])  # training accuracy
    axes[0].plot(data[:, 2])  # testing accuracy
    axes[0].legend(['Training', 'Testing'])
    axes[0].set_title('Accuracy Over Time')
    axes[0].set_xlabel('epoch')
    axes[0].set_ybound(0.0, 1.0)

    # same plot zoomed into [0.85, 1.00]
    axes[1].plot(np.log(1 - data[:, 0]))  # training accuracy
    axes[1].plot(np.log(1 - data[:, 2]))  # testing accuracy
    axes[1].legend(['Training', 'Testing'])
    axes[1].set_title('Log-Inverse Accuracy')
    axes[1].set_xlabel('epoch')
    # axes[1].set_ybound(0.90,1.0)
    plt.show()

if __name__ == '__main__':

    data, labels = preprocess_data('data')
    train(data, labels)
    calculate_accuracy('training.csv')


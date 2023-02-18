import argparse

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import optimizers, Sequential
from keras.layers import Conv2D
from keras.layers import Dense, MaxPooling2D, Flatten, Dropout
from keras.utils import to_categorical
from keras.utils.version_utils import callbacks
from sklearn.model_selection import train_test_split

from utils import preprocess_data


def train_ResNet50(x_train, x_test, y_train, y_test):
    y_train = to_categorical(y_train, 5, dtype='float32')
    y_test = to_categorical(y_test, 5, dtype='float32')

    from keras.applications.resnet_v2 import ResNet50V2
    from keras.applications.resnet_v2 import preprocess_input

    base_model = ResNet50V2(weights="imagenet", include_top=False, input_shape=x_train[0].shape)
    base_model.trainable = False
    x_train = preprocess_input(x_train)
    x_test = preprocess_input(x_test)

    model = tf.keras.models.Sequential([
        base_model,
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(5, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adamax(lr=0.001),
                  metrics=['accuracy'])

    csv_logger = tf.keras.callbacks.CSVLogger('ResNet50_training.csv', append=True)
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    model.fit(x_train, y_train,
              epochs=100,
              validation_data=[x_test, y_test],
              callbacks=[
                  csv_logger, early_stopping
              ])

    model.save("font_class_ResNet50.h5")
    calculate_accuracy('ResNet50_training.csv')


def train_VGG19(x_train, x_test, y_train, y_test):
    y_train = to_categorical(y_train, 5, dtype='float32')
    y_test = to_categorical(y_test, 5, dtype='float32')

    from keras.applications.vgg19 import VGG19
    from keras.applications.vgg19 import preprocess_input

    base_model = VGG19(weights="imagenet", include_top=False, input_shape=x_train[0].shape)
    base_model.trainable = False
    x_train = preprocess_input(x_train)
    x_test = preprocess_input(x_test)

    model = tf.keras.models.Sequential([
        base_model,
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(5, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adamax(lr=0.001),
                  metrics=['accuracy'])

    csv_logger = tf.keras.callbacks.CSVLogger('VGG19_training.csv', append=True)
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    model.fit(x_train, y_train,
              epochs=100,
              workers=10,
              validation_data=[x_test, y_test],
              callbacks=[
                  csv_logger, early_stopping
              ])

    model.save("font_class_VGG19.h5")
    calculate_accuracy('VGG19_training.csv')


def train_basic(x_train, x_test, y_train, y_test):
    y_train = to_categorical(y_train, 5, dtype='float32')
    y_test = to_categorical(y_test, 5, dtype='float32')

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=x_train[0].shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop(),
                  metrics=['accuracy'])

    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    csv_logger = tf.keras.callbacks.CSVLogger('training_basic.csv', append=True)

    x_test = x_test
    model.fit(x_train, y_train,
              batch_size=128,
              epochs=100,
              validation_data=[x_test, y_test],
              callbacks=[
                  csv_logger,
                  early_stopping
              ])

    model.save("font_class_basic.h5")
    calculate_accuracy('training_basic.csv')


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


def train_meta_model(train, test):
    basic = tf.keras.models.load_model('font_class_basic.h5')
    vgg19 = tf.keras.models.load_model('font_class_VGG19.h5')
    resnet50 = tf.keras.models.load_model('font_class_ResNet50.h5')

    basic_preds = basic.predict(train, verbose=2)
    vgg19_preds = vgg19.predict(train, verbose=2)
    resnet50_preds = resnet50.predict(train, verbose=2)

    ensemble_preds = [basic_preds, vgg19_preds, resnet50_preds]

    meta_features = np.column_stack(ensemble_preds)

    meta_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(5, activation='softmax', input_shape=meta_features[0].shape)
    ])
    meta_model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    early_stopping = callbacks.EarlyStopping(
        monitor="loss", patience=10, restore_best_weights=True
    )
    csv_logger = tf.keras.callbacks.CSVLogger('training_meta.csv', append=True)
    meta_model.fit(meta_features, test, batch_size=128, epochs=50, callbacks=[
        csv_logger,
        early_stopping
    ])
    meta_model.save("meta_model.h5")


def train(x_train, x_test, y_train, y_test):
    train_ResNet50(x_train, x_test, y_train, y_test)
    train_VGG19(x_train, x_test, y_train, y_test)
    train_basic(x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A model for font classification')
    parser.add_argument('--images_file', '-i', type=str, help='Path to the dataset file, must be .h5')
    parser.add_argument('--dataset_file', '-d', type=str, default='dataset', help='Path to the processed data')
    parser.add_argument('--test_size', '-t', type=float, default=0.2, help='Size of the training set')
    args = parser.parse_args()
    images_file = args.images_file
    dataset_file = args.dataset_file
    test_size = args.test_size

    preprocess_data(training_file=images_file,
                    out_file=dataset_file,
                    with_labels=False)

    print(f"Loading dataset file: {dataset_file}")
    dataset = np.load(dataset_file)
    data = dataset['x']
    labels = dataset['y']
    print(f"Loaded {len(data)} items")
    print(f"Data shape: {len(dataset['x'][0].shape)}")
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=42)
    train_meta_model(data, labels)
    print(f"Training finished successfully!")

import argparse
import csv
import re
from itertools import chain

import h5py
import numpy as np
import tensorflow as tf

from utils import preprocess_data, from_categorical


def predict(data):
    basic = tf.keras.models.load_model('font_class_basic.h5')
    vgg19 = tf.keras.models.load_model('font_class_VGG19.h5')
    resnet50 = tf.keras.models.load_model('font_class_ResNet50.h5')
    meta_model = tf.keras.models.load_model('meta_model.h5')

    basic_preds = basic.predict(data, verbose=2)
    vgg19_preds = vgg19.predict(data, verbose=2)
    resnet50_preds = resnet50.predict(data, verbose=2)
    ensemble_preds = [basic_preds, vgg19_preds, resnet50_preds]

    meta_features = np.column_stack(ensemble_preds)
    return meta_model.predict(meta_features)


def write_predictions(prediction_file: str, chars_img_names: [[]], predictions: list):
    labels = from_categorical(predictions)

    with open(prediction_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Name', 'WordIndex', 'CharacterIndex', 'Font'])

        for i in range(len(chars_img_names)):
            chars_img_names[i].append(labels[i])
            name, word_i, char_i, label = chars_img_names[i]
            writer.writerow([name, word_i, char_i, label])

    print(f"Prediction finished successfully. Result file: {prediction_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A model for font classification')
    parser.add_argument('--images_file', '-i', type=str, help='Path to the dataset file, must be .h5')
    parser.add_argument('--dataset_file', '-d', type=str, default='dataset.npz', help='Path to the processed data')
    parser.add_argument('--prediction_file', '-o', type=str, default='predictions.csv', help='Path to the output file')
    parser.add_argument('--save_cropped', '-s', type=bool, default=False, help='Should save the cropped images under '
                                                                               'cropped_images')
    args = parser.parse_args()
    images_file = args.images_file
    dataset_file = args.dataset_file
    prediction_file = args.prediction_file
    save_cropped = args.save_cropped

    chars_img_names = preprocess_data(training_file=images_file,
                                      out_file=dataset_file,
                                      with_labels=False,
                                      save_images=save_cropped)

    print(f"Loading dataset file: {dataset_file}")
    dataset = np.load(dataset_file)
    data = dataset['x']
    labels = dataset['y']
    print(f"Loaded {len(data)} items")
    print(f"Data shape: {len(dataset['x'][0].shape)}")

    test_data = np.load(dataset_file)['x']
    predictions = predict(test_data)

    write_predictions(prediction_file, chars_img_names, predictions)

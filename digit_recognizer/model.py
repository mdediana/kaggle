import argparse
import os
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD


IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
NUM_CHANNELS = 1
NUM_CLASSES = 10
EPOCHS = 200
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.1
MC_SAMPLES = 100
TENSORBOARD_LOG_DIR = os.path.join(os.curdir, 'logs/tensorboard', datetime.now().isoformat())

logger = logging.getLogger(__name__)


class MCDropout(Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)


def _read_csv(filename, split_X_y=True):
    """Return X, y if training data or X, None if test data"""
    df = pd.read_csv(filename)
    if split_X_y and 'label' in df:
        X = df.drop(columns=['label'])
        y = to_categorical(df['label'])
    else:
        X = df
        y = None
    X /= 255.0   # Scale pixels (min 0, max 255)
    X = X.values.reshape(len(X), IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS)  # X turned into a ndarray
    return X, y


def _read_data(training_set_file, test_set_file, is_2d=False):
    X_train, y_train = _read_csv(training_set_file, is_2d)
    X_test, _ = _read_csv(test_set_file, is_2d)
    logger.info('X train shape: %s', X_train.shape)
    logger.info('X test shape: %s', X_test.shape)
    return X_train, y_train, X_test


def _build_model(learning_rate=3e-3):  # Score: 0.99071
    model = Sequential([
        Conv2D(filters=64, kernel_size=7, strides=1, padding='same', activation='relu',
               input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS)),
        MaxPooling2D(pool_size=2),
        Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
        Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
        MaxPooling2D(pool_size=2),
        Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
        Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
        MaxPooling2D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        MCDropout(0.5),
        Dense(64, activation='relu'),
        MCDropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    optimizer = SGD(lr=learning_rate, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[SCORING])
    model.summary()
    return model


def _build_model_tuner(hp):
    n_neurons = [hp.Int('units_' + str(i), min_value=1, max_value=512, step=32)
                 for i in range(hp.Int('num_layers', 0, 5))]
    learning_rate = hp.Choice('learning_rate', [1e-3, 3e-3, 5e-3])
    dropout_rate = hp.Choice('dropout_rate', [0.05, 0.1, 0.15])
    return _build_model_dense(n_neurons, learning_rate, dropout_rate)


def train(training_set_file, test_set_file, model_file=None, epochs=EPOCHS, batch_size=BATCH_SIZE,
          validation_split=VALIDATION_SPLIT):
    X_train, y_train, _ = _read_data(training_set_file, test_set_file, is_2d=True)
    model = _build_model()
    callbacks = [TensorBoard(TENSORBOARD_LOG_DIR), EarlyStopping(patience=10)]
    model.fit(X_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_split=validation_split,
              verbose=2,
              callbacks=callbacks)
    if model_file is not None:
        logger.info('Saving model to file: %s', model_file)
        model.save(model_file)
    return model


def predict(training_set_file, test_set_file, model_file=None, output_file=None):
    _, _, X_test = _read_data(training_set_file, test_set_file, is_2d=True)
    if model_file is None:
        logger.info('Building and training model')
        model = train(training_set_file, test_set_file, model_file)
    else:
        logger.info('Reading model from file: %s', model_file)
        model = load_model(model_file, custom_objects={'MCDropout': MCDropout})
        model.summary()
    logger.info('Predict using mode of %d Monte Carlo runs', MC_SAMPLES)
    ys = np.stack([model.predict_classes(X_test) for i in range(MC_SAMPLES)])
    y = stats.mode(ys).mode.flatten()
    out_df = pd.DataFrame({"ImageId": range(1, len(y) + 1), "Label": y})
    if output_file is not None:
        out_df.to_csv(output_file, index=False)
    else:
        logger.info('No output file set, number of predictions: %d', len(out_df))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recognize digits in the MNIST dataset')
    parser.add_argument('command', help='Command to run', choices=['train', 'predict', 'search-params'],
                        default='train')
    parser.add_argument('--training-set-file', help='Training set file', default='train.csv')
    parser.add_argument('--test-set-file', help='Test set file', default='test.csv')
    parser.add_argument('--output-file', help='Output file')
    parser.add_argument('--model-file', help='Model file, needs to have h5 file extension')
    args = parser.parse_args()

    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if args.model_file is not None and not args.model_file.endswith('.h5'):
        parser.error('Model file needs to have h5 file extension')
    if args.command == 'train':
        train(args.training_set_file, args.test_set_file, args.model_file)
    elif args.command == 'predict':
        predict(args.training_set_file, args.test_set_file, args.model_file, args.output_file)

import argparse
import os
import logging
from datetime import datetime

import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense


N_JOBS = -1  # Use all processors, particularly useful when param grid searching
CV_FOLDS = 5
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
NUMBER_OF_CLASSES = 10
SCORING = 'accuracy'
TENSORBOARD_LOG_DIR = os.path.join(os.curdir, 'logs', datetime.now().isoformat())
PARAM_GRIDS = {
}
BEST_PARAMS = {
}

logger = logging.getLogger(__name__)


def _read_csv(filename, split_X_y=True):
    """Return X, y if training data or X, None if test data"""
    df = pd.read_csv(filename)
    if split_X_y and 'label' in df:
        X = df.drop(columns=['label'])
        y = to_categorical(df['label'])
    else:
        X = df
        y = None
    # Scale pixels (min 0, max 255)
    X = X / 255.0
    return X, y


def _read_data(training_set_file, test_set_file):
    X_train, y_train = _read_csv(training_set_file)
    X_test, _ = _read_csv(test_set_file)
    logger.info('X shape: %s', X_train.shape)
    return X_train, y_train, X_test


def _build_model(use_best_params=True):
    model = Sequential()
    model.add(Flatten(input_shape=[IMAGE_HEIGHT * IMAGE_WIDTH]))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(NUMBER_OF_CLASSES, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=[SCORING])
    return model


def _train(X, y, model):
    tensorboard_cb = TensorBoard(TENSORBOARD_LOG_DIR)
    model.fit(X, y, epochs=10, batch_size=16, validation_split=0.1, verbose=2, callbacks=[tensorboard_cb])


def _predict(model, X):
    y = model.predict_classes(X, verbose=0)
    return pd.DataFrame({"ImageId": range(1, len(y) + 1), "Label": y})


def train(training_set_file, test_set_file):
    X_train, y_train, _ = _read_data(training_set_file, test_set_file)
    clf = _build_model()
    _train(X_train, y_train, clf)


def predict(training_set_file, test_set_file, output_file):
    X_train, y_train, X_test = _read_data(training_set_file, test_set_file)
    model = _build_model()
    _train(X_train, y_train, model)
    y = _predict(model, X_test)
    y.to_csv(output_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recognize digits in the MNIST dataset')
    parser.add_argument('command', help='Command to run', choices=['train', 'predict', 'search-params'],
                        default='train')
    parser.add_argument('--training-set-file', help='Training set file', default='train.csv')
    parser.add_argument('--test-set-file', help='Test set file', default='test.csv')
    parser.add_argument('--output-file', help='Output file', default='output.csv')
    args = parser.parse_args()

    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if args.command == 'train':
        train(args.training_set_file, args.test_set_file)
    elif args.command == 'predict':
        predict(args.training_set_file, args.test_set_file, args.output_file)
    # elif args.command == 'search-params':
    #     search_params(args.training_set_file, args.test_set_file)

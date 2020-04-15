import argparse
import os
import sys
import logging
from datetime import datetime

import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import SGD
from kerastuner.tuners import RandomSearch


IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
NUMBER_OF_CLASSES = 10
SCORING = 'accuracy'
TENSORBOARD_LOG_DIR = os.path.join(os.curdir, 'logs/tensorboard', datetime.now().isoformat())
KERASTUNER_DIR = os.path.join(os.curdir, 'logs/kerastuner')
BEST_PARAMS = dict(
    n_neurons=[449, 33],  # Neurons on each hidden layer
    learning_rate=0.001
)

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
    logger.info('X train shape: %s', X_train.shape)
    logger.info('X test shape: %s', X_test.shape)
    return X_train, y_train, X_test


def _build_model(n_neurons=[30], learning_rate=3e-3):
    model = Sequential()
    model.add(InputLayer(input_shape=[IMAGE_HEIGHT * IMAGE_WIDTH]))
    for n in n_neurons:
        model.add(Dense(n, activation='relu'))
    model.add(Dense(NUMBER_OF_CLASSES, activation='softmax'))
    optimizer = SGD(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[SCORING])
    model.summary()
    return model


def _build_model_tuner(hp):
    model = Sequential()
    model.add(InputLayer(input_shape=[IMAGE_HEIGHT * IMAGE_WIDTH]))
    for i in range(hp.Int('num_layers', 0, 5)):
        model.add(Dense(
            units=hp.Int('units_' + str(i), min_value=1, max_value=512, step=32),
            activation='relu'))
    model.add(Dense(NUMBER_OF_CLASSES, activation='softmax'))
    optimizer = SGD(lr=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4]))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[SCORING])
    model.summary()
    return model


def train(training_set_file, test_set_file, model_file=None):
    X_train, y_train, _ = _read_data(training_set_file, test_set_file)
    model = _build_model(**BEST_PARAMS)
    callbacks = [TensorBoard(TENSORBOARD_LOG_DIR), EarlyStopping(patience=10)]
    model.fit(X_train, y_train,
              epochs=100,
              batch_size=16,
              validation_split=0.1,
              verbose=2,
              callbacks=callbacks)
    if model_file is not None:
        logger.info('Saving model to file: %s', model_file)
        model.save(model_file)
    return model


def predict(training_set_file, test_set_file, model_file=None, output_file=sys.stdout):
    _, _, X_test = _read_data(training_set_file, test_set_file)
    if model_file is None:
        logger.info('Building and training model')
        model = train(training_set_file, test_set_file, model_file)
    else:
        # TODO: Fix this
        logger.info('Reading model from file: %s', model_file)
        model = load_model(model_file)
        model.summary()
    y = model.predict_classes(X_test, verbose=0)
    out_df = pd.DataFrame({"ImageId": range(1, len(y) + 1), "Label": y})
    out_df.to_csv(output_file, index=False)


def search_params(training_set_file, test_set_file):
    X_train, y_train, _ = _read_data(training_set_file, test_set_file)
    tuner = RandomSearch(_build_model_tuner,
                         objective='val_accuracy',
                         max_trials=5,
                         executions_per_trial=2,
                         directory=KERASTUNER_DIR,
                         project_name='mnist')
    tuner.search_space_summary()
    tuner.search(X_train, y_train, epochs=10, batch_size=16, validation_split=0.1, verbose=2)
    tuner.results_summary()
    model = tuner.get_best_models(num_models=1)[0]
    logger.info('Best model: ')
    model.summary()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recognize digits in the MNIST dataset')
    parser.add_argument('command', help='Command to run', choices=['train', 'predict', 'search-params'],
                        default='train')
    parser.add_argument('--training-set-file', help='Training set file', default='train.csv')
    parser.add_argument('--test-set-file', help='Test set file', default='test.csv')
    parser.add_argument('--output-file', help='Output file', default='output.csv')
    parser.add_argument('--model-file', help='Model file')
    args = parser.parse_args()

    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if args.command == 'train':
        train(args.training_set_file, args.test_set_file, args.model_file)
    elif args.command == 'predict':
        predict(args.training_set_file, args.test_set_file, args.model_file, args.output_file)
    elif args.command == 'search-params':
        search_params(args.training_set_file, args.test_set_file)

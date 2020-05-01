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
from tensorflow.keras.layers import Flatten, InputLayer, Dense, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from kerastuner.tuners import RandomSearch


NUM_INSTANCES = 42000
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
NUM_CHANNELS = 1
PNG_DTYPE = tf.uint8
NUM_CLASSES = 10
NUM_EPOCHS = 200
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.1
SCORING = 'accuracy'
ACTIVATION = 'selu'
KERNEL_INITIALIZER = 'lecun_uniform'
MC_SAMPLES = 100
TENSORBOARD_LOG_DIR = os.path.join(os.curdir, 'logs/tensorboard', datetime.now().isoformat())
KERASTUNER_DIR = os.path.join(os.curdir, 'logs/kerastuner')
BEST_PARAMS = dict(
    n_neurons=[385, 257, 449],  # Neurons on each hidden layer
    learning_rate=0.003,
    dropout_rate=0.1,
)

logger = logging.getLogger(__name__)


class MCDropout(Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)


def _read_test_csv(filename):
    X = pd.read_csv(filename)
    X /= 255.0   # Scale pixels (min 0, max 255)
    X = X.values.reshape(len(X), IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS)  # X turned into a ndarray
    return X


def _read_csv(filename, is_2d=True, split_X_y=True):
    """Return X, y if training data or X, None if test data"""
    df = pd.read_csv(filename)
    if split_X_y and 'label' in df:
        X = df.drop(columns=['label'])
        y = to_categorical(df['label'])
    else:
        X = df
        y = None
    # X /= 255.0   # Scale pixels (min 0, max 255)
    X = tf.dtypes.cast(X, tf.uint16)
    if is_2d:
        X = X.values.reshape(len(X), IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS)  # X turned into a ndarray
    else:
        X = X.values
    return X, y


def _read_tfrecords(training_set_file, ds_type=None):
    description = {
        'png': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    def _parse_and_normalize(record):
        example = tf.io.parse_single_example(record, description)
        png = tf.image.decode_png(example['png'], dtype=PNG_DTYPE)
        normalized = tf.math.divide(png, 255)
        categories = tf.one_hot(example['label'], depth=NUM_CLASSES)
        return normalized, categories

    num_train = int((1 - VALIDATION_SPLIT) * NUM_INSTANCES)
    ds = tf.data.TFRecordDataset([training_set_file]).map(_parse_and_normalize)
    if ds_type == 'train':
        logger.info('Number of training instances: %d of %d', num_train, NUM_INSTANCES)
        ds = ds.take(num_train)
    elif ds_type == 'validation':
        logger.info('Number of validation instances: %d of %d', NUM_INSTANCES - num_train, NUM_INSTANCES)
        ds = ds.skip(num_train)
    else:
        logger.info('Full dataset, number of instances: %d', NUM_INSTANCES)
    return ds \
        .batch(BATCH_SIZE) \
        .cache() \
        .shuffle(NUM_INSTANCES, seed=0)  # This is a small dataset, a large buffer is not a problem


def _build_model(learning_rate=3e-3):  # Score: 0.99171 (shuffle datasets)
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


def train(training_set_file, test_set_file, model_file=None):
    ds_train = _read_tfrecords(training_set_file, ds_type='train')
    ds_val = _read_tfrecords(training_set_file, ds_type='validation')
    model = _build_model()
    callbacks = [TensorBoard(TENSORBOARD_LOG_DIR), EarlyStopping(patience=10)]
    model.fit(x=ds_train,
              validation_data=ds_val,
              epochs=NUM_EPOCHS,
              batch_size=BATCH_SIZE,
              verbose=2,
              callbacks=callbacks)
    if model_file is not None:
        logger.info('Saving model to file: %s', model_file)
        model.save(model_file)
    logger.info('Done')
    return model


def predict(training_set_file, test_set_file, model_file=None, output_file=None):
    if model_file is None:
        logger.info('Building and training model')
        model = train(training_set_file, test_set_file, model_file)
    else:
        logger.info('Reading model from file: %s', model_file)
        model = load_model(model_file, custom_objects={'MCDropout': MCDropout})
        model.summary()
    X_test = _read_test_csv(test_set_file)
    logger.info('X test shape: %s', X_test.shape)
    logger.info('Predict using mode of %d Monte Carlo runs', MC_SAMPLES)
    # TODO: WARNING:tensorflow:From model.py:186: Sequential.predict_classes (from
    # tensorflow.python.keras.engine.sequential) is deprecated and will be removed after 2021-01-01.
    # Instructions for updating:
    # Please use instead:* `np.argmax(model.predict(x), axis=-1)`,   if your model does multi-class classification
    # (e.g. if it uses a `softmax` last-layer activation).* `(model.predict(x) > 0.5).astype("int32")`,
    # if your model does binary classification   (e.g. if it uses a `sigmoid` last-layer activation).
    ys = np.stack([model.predict_classes(X_test) for i in range(MC_SAMPLES)])  # TODO: Use loop to print progress
    y = stats.mode(ys).mode.flatten()
    out_df = pd.DataFrame({"ImageId": range(1, len(y) + 1), "Label": y})
    if output_file is not None:
        out_df.to_csv(output_file, index=False)
    else:
        logger.info('No output file set, number of predictions: %d', len(out_df))
    logger.info('Done')


def search_params(training_set_file, test_set_file):
    X_train, y_train, _ = _read_data(training_set_file, test_set_file)
    tuner = RandomSearch(_build_model_tuner,
                         objective='val_accuracy',
                         max_trials=5,
                         executions_per_trial=2,
                         directory=KERASTUNER_DIR,
                         project_name='mnist')
    tuner.search_space_summary()
    tuner.search(X_train, y_train,
                 epochs=NUM_EPOCHS,
                 batch_size=BATCH_SIZE,
                 validation_split=VALIDATION_SPLIT,
                 verbose=2)
    tuner.results_summary()
    model = tuner.get_best_models(num_models=1)[0]
    logger.info('Best model: ')
    model.summary()


def _bytes_feature(value):
    # From https://www.tensorflow.org/tutorials/load_data/tfrecord#tfexample
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    # From https://www.tensorflow.org/tutorials/load_data/tfrecord#tfexample
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def csv_to_tfrecords(input_file, output_file):
    dataset = tf.data.experimental.make_csv_dataset(
        input_file,
        label_name='label',
        batch_size=1,
        num_epochs=1,   # Single pass over the dataset
        shuffle=False)  # Keep the same order as the CSV
    logger.info('Starting conversion')
    with tf.io.TFRecordWriter(output_file) as writer:
        for i, record in enumerate(dataset, 1):
            if i % 100 == 0:
                logger.info('Records processed: %d', i)
            x, y = record
            x = tf.cast(tf.stack(list(x.values()), axis=-1), PNG_DTYPE)  # Must match the decode_png type
            png = tf.reshape(x, [IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'png': _bytes_feature(tf.image.encode_png(png)),
                    'label': _int64_feature(y)})
                )
            writer.write(example.SerializeToString())
    logger.info('Conversion done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recognize digits in the MNIST dataset')
    parser.add_argument('command', help='Command to run',
                        choices=['train', 'predict', 'search-params', 'csv-to-tfrecords'],
                        default='train')
    parser.add_argument('--training-set-file', help='Training set file', default='train.tf_records')
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
    elif args.command == 'search-params':
        search_params(args.training_set_file, args.test_set_file)
    elif args.command == 'csv-to-tfrecords':
        csv_to_tfrecords(args.training_set_file, args.output_file)

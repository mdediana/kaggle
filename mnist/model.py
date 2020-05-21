import argparse
import os
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD


NUM_INSTANCES = 42000
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
NUM_CHANNELS = 1
PNG_DTYPE = tf.uint8
PNG_TFRECORD = {
    'png': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
}
NUM_CLASSES = 10
NUM_EPOCHS = 200
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.1
MC_SAMPLES = 100
TENSORBOARD_LOG_DIR = os.path.join(os.curdir, 'logs/tensorboard', datetime.now().isoformat())
TENSORBOARD_IMG_DIR = os.path.join(os.curdir, 'logs/tensorboard/images', datetime.now().isoformat())


logger = logging.getLogger(__name__)


class MCDropout(Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)


def _read_test_csv(filename):
    X = pd.read_csv(filename)
    X /= 255.0   # Scale pixels (min 0, max 255)
    X = X.values.reshape(len(X), IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS)  # X turned into a ndarray
    return X


def _read_tfrecords(training_set_file, ds_type=None):
    def _parse_and_normalize(record):
        example = tf.io.parse_single_example(record, PNG_TFRECORD)
        png = tf.image.decode_png(example['png'], dtype=PNG_DTYPE)
        normalized = tf.math.divide(png, 255)
        categories = tf.one_hot(example['label'], depth=NUM_CLASSES)
        return normalized, categories

    if os.path.isfile(training_set_file):
        files = [training_set_file]
    else:
        files = [os.path.join(training_set_file, f) for f in os.listdir(training_set_file)]
    logger.info('Training set files: %s', files)
    num_instances = NUM_INSTANCES * len(files)  # Consider that all files contain the same number of instances
    num_train = int((1 - VALIDATION_SPLIT) * num_instances)
    ds = tf.data.TFRecordDataset(files).map(_parse_and_normalize)
    if ds_type == 'train':
        logger.info('Number of training instances: %d of %d', num_train, num_instances)
        ds = ds.take(num_train)
    elif ds_type == 'validation':
        logger.info('Number of validation instances: %d of %d', num_instances - num_train, num_instances)
        ds = ds.skip(num_train)
    else:
        logger.info('Full dataset, number of instances: %d', num_instances)
    return ds \
        .batch(BATCH_SIZE) \
        .cache() \
        .shuffle(num_instances, seed=0)  # This is a small dataset, a large buffer is not a problem


def _build_model(learning_rate=3e-3):
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
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model


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
    preds_i = list()
    for i in range(MC_SAMPLES):
        preds_i.append(np.argmax(model.predict(X_test), axis=-1))
        logger.info('Run finished: %d', i + 1)
    preds = stats.mode(np.stack(preds_i)).mode.flatten()
    out_df = pd.DataFrame({"ImageId": range(1, len(preds) + 1), "Label": preds})
    if output_file is not None:
        out_df.to_csv(output_file, index=False)
    else:
        logger.info('No output file set, number of predictions: %d', len(out_df))
    logger.info('Done')


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
    logger.info('Start conversion')
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


def _transform_image(png):  # Score: 0.99371
    new_size = tf.random.uniform(shape=[], minval=int(1/4*28), maxval=int(3/4*28), dtype=tf.dtypes.int32)
    offset_height = tf.random.uniform(shape=[], minval=0, maxval=int(28-new_size), dtype=tf.dtypes.int32)
    offset_width = tf.random.uniform(shape=[], minval=0, maxval=int(28-new_size), dtype=tf.dtypes.int32)
    png = tf.image.resize_with_pad(png, new_size, new_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    png = tf.image.pad_to_bounding_box(png, offset_height, offset_width, 28, 28)
    png = tf.image.random_brightness(png, max_delta=0.3)
    png = tf.image.random_contrast(png, lower=0.7, upper=1.5)
    return png


def augment_data(input_file, output_file):
    def _transform(record):
        example = tf.io.parse_single_example(record, PNG_TFRECORD)
        png = tf.image.decode_png(example['png'], dtype=PNG_DTYPE)
        new_png = _transform_image(png)
        return new_png, example['label']

    logger.info('Start conversion')
    transformed = tf.data.TFRecordDataset([input_file]).map(_transform)
    with tf.io.TFRecordWriter(output_file) as writer:
        for i, record in enumerate(transformed, 1):
            if i % 100 == 0:
                logger.info('Records processed: %d', i)
            png, label = record
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    'png': _bytes_feature(tf.image.encode_png(png)),
                    'label': _int64_feature(label)})
            )
            writer.write(example.SerializeToString())
    logger.info('Conversion done')


def _debug_images(imgs, name='Training data'):
    imgs = np.reshape(list(imgs), (-1, 28, 28, 1))
    file_writer = tf.summary.create_file_writer(TENSORBOARD_IMG_DIR)
    with file_writer.as_default():
        tf.summary.image(name, imgs, max_outputs=100, step=0)


def test_augment_data(input_file, num_images=20):
    def _transform(record):
        example = tf.io.parse_single_example(record, PNG_TFRECORD)
        png = tf.image.decode_png(example['png'], dtype=PNG_DTYPE)
        return _transform_image(png)

    imgs = tf.data.TFRecordDataset([input_file]).take(num_images).map(_transform)
    _debug_images(imgs)


def debug_images(input_file, num_images=20):
    def _parse(record):
        example = tf.io.parse_single_example(record, PNG_TFRECORD)
        return tf.image.decode_png(example['png'], dtype=PNG_DTYPE)

    imgs = tf.data.TFRecordDataset([input_file]).take(num_images).map(_parse)
    _debug_images(imgs, name=input_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recognize digits in the MNIST dataset')
    parser.add_argument('command', help='Command to run',
                        choices=['train', 'predict', 'csv-to-tfrecords', 'augment-data', 'debug-images'],
                        default='train')
    parser.add_argument('--training-set-file', help='Training set file or directory', default='train.tf_records')
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
    elif args.command == 'csv-to-tfrecords':
        csv_to_tfrecords(args.training_set_file, args.output_file)
    elif args.command == 'augment-data':
        augment_data(args.training_set_file, args.output_file)
        # test_augment_data(args.training_set_file)
    elif args.command == 'debug-images':
        debug_images(args.training_set_file)

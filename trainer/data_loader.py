# Some portion of the code has been referred from here:
# https://bit.ly/2UX6Dpx

import tensorflow as tf
import numpy as np
import config
import re


def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)


def read_tfrecord(example):
	features = {
		"image": tf.io.FixedLenFeature([], tf.string),
		"class": tf.io.FixedLenFeature([], tf.int64)
	}
	example = tf.io.parse_single_example(example, features)
	image = tf.image.decode_jpeg(example["image"], channels=3)
	image = tf.reshape(image,
					   [config.IMAGE_SIZE, config.IMAGE_SIZE, 3])
	class_label = tf.cast(example["class"], tf.int32)
	return image, class_label


def load_dataset(filenames):
	options_no_order = tf.data.Options()
	options_no_order.experimental_deterministic = False
	dataset = tf.data.TFRecordDataset(filenames,
									  num_parallel_reads=config.AUTO)
	dataset = dataset.with_options(options_no_order)
	dataset = dataset.map(read_tfrecord,
						  num_parallel_calls=config.AUTO)
	return dataset


def batch_dataset(filenames, batch_size, train=True):
	dataset = load_dataset(filenames)
	if train:
		dataset = dataset.repeat()
		dataset = dataset.shuffle(batch_size * 100)
	dataset = dataset.batch(batch_size)
	dataset = dataset.prefetch(config.AUTO)
	return dataset


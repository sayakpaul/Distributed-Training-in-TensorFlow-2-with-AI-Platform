from tqdm import tqdm
import tensorflow as tf
import config
import os

def augment(image):
	image = tf.image.random_flip_left_right(image)
	image = tf.image.random_flip_up_down(image)

	if tf.random.uniform([], tf.float32) < 0.5:
		image = tf.image.rot90(image)
	return image


def preprocess_image(image, label, train=True):
	image = tf.image.resize(image,
							(config.IMAGE_SIZE, config.IMAGE_SIZE))
	if train:
		image = augment(image)
	image = tf.cast(image, tf.uint8)
	image = tf.image.encode_jpeg(image, optimize_size=True,
								 chroma_downsampling=False)
	return image, label


def prepare_dataset_tfr(dataset: tf.data.Dataset, train=True):
	dataset = dataset.map(lambda x, y: (preprocess_image(x, y, train)),
						  num_parallel_calls=config.AUTO)
	dataset = dataset.batch(config.SHARD_SIZE)
	return dataset


def bytestring_feature(list_of_bytestrings):
	return tf.train.Feature(
		bytes_list=tf.train.BytesList(value=list_of_bytestrings))


def int_feature(list_of_ints):
	return tf.train.Feature(
		int64_list=tf.train.Int64List(value=list_of_ints))


def to_tfrecord(img_bytes, label):
	feature = {
		"image": bytestring_feature([img_bytes]),
		"class": int_feature([label]),
	}
	return tf.train.Example(
		features=tf.train.Features(feature=feature))


def write_tfrecords(dataset: tf.data.Dataset, output_dir, print_every=5):
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	for shard, (image, label) in enumerate(tqdm(dataset)):
		shard_size = image.numpy().shape[0]
		filename = output_dir + "/catsdogs-" + "{:02d}-{}.tfrec".format(shard, shard_size)

		with tf.io.TFRecordWriter(filename) as out_file:
			for i in range(shard_size):
				example = to_tfrecord(image.numpy()[i],
									  label.numpy()[i])
				out_file.write(example.SerializeToString())
			if shard % print_every == 0:
				print("Wrote file {} containing {} records".format(
						filename, shard_size))
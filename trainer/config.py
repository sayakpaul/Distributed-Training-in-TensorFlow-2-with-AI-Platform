import tensorflow as tf

# Data-related variables
IMAGE_SIZE = 224
SHARD_SIZE = 128
AUTO = tf.data.AUTOTUNE
TRAIN_TFR = "train_tfr"
VALID_TFR = "validation_tfr"
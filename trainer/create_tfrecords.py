# Referred from the following notebook:
# https://github.com/sayakpaul/TF-2.0-Hacks/blob/master/Cats_vs_Dogs_TFRecords.ipynb

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import tfr_utils
import config

# Load the cats-dogs dataset
print("Loading dataset from TensorFlow Datasets")
train_ds, validation_ds = tfds.load(
    "cats_vs_dogs",
    split=["train[:90%]", "train[90%:]"],
    as_supervised=True
)

# Prepare the datasets for serialization
train_ds = tfr_utils.prepare_dataset_tfr(train_ds)
validation_ds = tfr_utils.prepare_dataset_tfr(validation_ds, train=False)

# Serialize as TFRecords
print("Serializing as TFRecords")
tfr_utils.write_tfrecords(train_ds, config.TRAIN_TFR)
tfr_utils.write_tfrecords(validation_ds, config.VALID_TFR)
from tensorflow import  keras
import tensorflow as tf
import data_loader
import datetime
import argparse
import config

# Construct argument parser
args_parser = argparse.ArgumentParser()
args_parser.add_argument(
        "--bucket",
        help="Name of the GCS Bucket,",
        required=True)
args_parser.add_argument(
        "--train-pattern",
        help="Pattern of GCS paths to training data.",
        required=True)
args_parser.add_argument(
        "--valid-pattern",
        help="Pattern of GCS paths to validation data.",
        required=True)
args_parser.parse_args()

# Detect training strategy
try:
	tpu = None
	tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
	tf.config.experimental_connect_to_cluster(tpu)
	tf.tpu.experimental.initialize_tpu_system(tpu)
	strategy = tf.distribute.TPUStrategy(tpu)
except ValueError: # detect GPUs
	strategy = tf.distribute.MirroredStrategy()

print("Number of accelerators: ", strategy.num_replicas_in_sync)
BATCH_SIZE = 128 * strategy.num_replicas_in_sync

# Load data
train_files = tf.io.gfile.glob(args_parser["train_pattern"])
validation_files = tf.io.gfile.glob(args_parser["valid_pattern"])
train_ds = data_loader.batch_dataset(train_files, BATCH_SIZE)
validation_ds = data_loader.batch_dataset(validation_files, BATCH_SIZE, False)

# Construct base model
base_model = keras.applications.MobileNetV3Small(
	weights="imagenet",
	input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3),
	include_top=False,
	pooling="avg"
)
base_model.trainable = False

# Construct classification top
inputs = keras.Input(shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3))
x = keras.layers.experimental.preprocessing.Rescaling(scale=1./127.5, offset=-1)(inputs)
x = base_model(x, training=False)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

# Set up callbacks
es = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau()
tb = keras.callbacks.TensorBoard(log_dir="gs://" + args_parser["bucket"] + "/logs")

# Linear transfer
model.compile(
	optimizer="adam",
	loss=keras.losses.BinaryCrossentropy(from_logits=True),
	metrics=[keras.metrics.BinaryAccuracy()],
)
model.fit(train_ds,
	steps_per_epoch=data_loader.count_data_items(train_files)//BATCH_SIZE,
	validation_data=validation_ds,
	epochs=100,
	callbacks=[es, tb]
)

# Fine-tuning
base_model.trainable = True
model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)
model.fit(train_ds,
	steps_per_epoch=data_loader.count_data_items(train_files)//BATCH_SIZE,
	validation_data=validation_ds,
	epochs=100,
	callbacks=[es, reduce_lr, tb]
)

# Serialize model
model_name = datetime.datetime.now().strftime("cats_dogs_%Y%m%d_%H%M%S")
model.save("gs://" + args_parser["bucket"] + "/" + model_name)

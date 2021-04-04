from tensorflow import keras
import tensorflow as tf
import data_loader
import model_utils
import datetime
import argparse

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
print(f"Using a batch size of {BATCH_SIZE}")

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
args = args_parser.parse_args()

# Load data
train_files = tf.io.gfile.glob(args["train_pattern"])
validation_files = tf.io.gfile.glob(args["valid_pattern"])
train_ds = data_loader.batch_dataset(train_files, BATCH_SIZE)
validation_ds = data_loader.batch_dataset(validation_files, BATCH_SIZE, False)

# Set up callbacks
es = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau()
log_dir = datetime.datetime.now().strftime("logs_%Y%m%d_%H%M%S")
tb = keras.callbacks.TensorBoard(log_dir="gs://" + args["bucket"] + "/" + log_dir)

# Linear transfer
with strategy.scope():
	model = model_utils.get_model()
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
with strategy.scope():
	for layer in model.layers[-30:]:
		if not isinstance(layer, keras.layers.BatchNormalization):
			layer.trainable = True
	model.compile(
		optimizer=keras.optimizers.Adam(1e-5 * strategy.num_replicas_in_sync),
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
model.save("gs://" + args["bucket"] + "/" + model_name)

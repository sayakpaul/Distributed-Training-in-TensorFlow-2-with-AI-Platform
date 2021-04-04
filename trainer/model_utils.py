from tensorflow import keras
import config


def get_model():
    # Construct base model
    base_model = keras.applications.DenseNet121(
        weights="imagenet",
        input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3),
        include_top=False,
        pooling="avg",
    )
    base_model.trainable = False
    # Construct classification top
    inputs = keras.Input(shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3))
    x = keras.applications.densenet.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)

    return model

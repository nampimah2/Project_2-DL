"""CNN architectures for CIFAR-100. Images are NHWC float32 in [0, 1]."""
import tensorflow as tf


def build_model(architecture: str, learning_rate: float = 1e-3,
               dropout: float = 0.35, l2_strength: float = 1e-4) -> tf.keras.Model:
    """Build and compile one of two CNNs using one-hot labels.

    Both models contain at least two convolution + max-pooling stages, a
    connected hidden layer, and a 100-way softmax output. L2 kernel penalties
    are applied throughout; dropout is applied before the classifier.
    """
    reg = tf.keras.regularizers.l2(l2_strength)
    inputs = tf.keras.Input(shape=(32, 32, 3), name="image")
    x = inputs
    if architecture == "compact":
        x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False,
                                   kernel_regularizer=reg)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False,
                                   kernel_regularizer=reg)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(dropout * 0.5)(x)
        x = tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False,
                                   kernel_regularizer=reg)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=reg)(x)
    elif architecture == "deep":
        for filters in (32, 64, 128):
            x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False,
                                       kernel_regularizer=reg)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False,
                                       kernel_regularizer=reg)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.MaxPooling2D()(x)
            x = tf.keras.layers.Dropout(dropout * 0.5)(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=reg)(x)
    else:
        raise ValueError("architecture must be 'compact' or 'deep'")
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(100, activation="softmax", name="class_probabilities")(x)
    model = tf.keras.Model(inputs, outputs, name=f"cifar100_{architecture}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
                 tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_accuracy")],
    )
    return model

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models

# Load EMNIST Balanced dataset
(ds_train, ds_test), ds_info = tfds.load(
    "emnist/balanced",
    split=["train", "test"],
    as_supervised=True,
    with_info=True
)

NUM_CLASSES = ds_info.features["label"].num_classes
print("Number of classes:", NUM_CLASSES)  # Expect 47


# Preprocessing function
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0

    # Fix EMNIST orientation
    image = tf.image.rot90(image, k=3)
    image = tf.image.flip_left_right(image)

    # DO NOT add channel dimension (already present!)
    return image, label




# Prepare datasets
ds_train = ds_train.map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)
ds_test  = ds_test.map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1)
])

# CNN model
model = tf.keras.Sequential([
    data_augmentation,

    layers.Conv2D(32, (3,3), padding="same", activation="relu", input_shape=(28,28,1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(NUM_CLASSES, activation="softmax")
])



# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model
model.fit(
    ds_train,
    validation_data=ds_test,
   epochs=20

)

# Save model
model.save("emnist_model.keras")
print("EMNIST model saved successfully!")

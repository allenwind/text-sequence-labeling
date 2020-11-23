import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

physical_devices = tf.config.experimental.list_physical_devices("GPU")
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


from layers import ConditionalRandomField

from dataset_cws import dataset, vocab_size

num_labels = 4

inputs = layers.Input(shape=(None,))
x = layers.Embedding(input_dim=vocab_size, output_dim=128, mask_zero=True)(inputs)
# x = layers.LSTM(128, return_sequences=True)(x)
x = layers.Conv1D(filters=128, kernel_size=2, padding="same", activation="relu")(x)
x = layers.Conv1D(filters=128, kernel_size=2, padding="same", activation="relu")(x)
x = layers.Conv1D(filters=128, kernel_size=2, padding="same", activation="relu")(x)
x = layers.Dense(num_labels)(x)
crf = ConditionalRandomField(lr_multiplier=1)
outputs = crf(x)

model = keras.Model(inputs, outputs)
model.summary()

model.compile(loss=crf.sparse_loss, optimizer="adam", metrics=[crf.sparse_accuracy])



model.fit(dataset, epochs=10, steps_per_epoch=300)

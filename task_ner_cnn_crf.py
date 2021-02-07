from tensorflow import keras
from tensorflow.keras import layers

from layers import CRF

inputs = layers.Input(shape=(None,), dtype="int32")
x = inputs
x = layers.Embedding(len(chars)+1, 128)(x)
x = layers.Conv1D(128, 3, activation="relu", padding="same")(x)
x = layers.Conv1D(128, 3, activation="relu", padding="same")(x)
x = layers.Conv1D(128, 3, activation="relu", padding="same")(x)
x = layers.Dense(4)(x)
crf = CRF(False)
outputs = crf(x)

model = keras.Model(inputs, outputs)
model.compile(loss=crf.loss, optimizer="adam", metrics=[crf.accuracy])
model.fit(dataset, epochs=10)

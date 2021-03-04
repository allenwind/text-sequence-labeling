import random
import collections
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import *
from tqdm import tqdm

physical_devices = tf.config.experimental.list_physical_devices("GPU")
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import dataset
from layers import MaskBiLSTM
from crf import CRF, ModelWithCRFLoss
from evaluation import Evaluator, evaluate_prf
from snippets import *

load_dataset = dataset.load_cpd
X_train, y_train, classes = load_dataset("train", with_labels=True)
id2label = {i:j for i,j in enumerate(classes)}
label2id = {j:i for i,j in id2label.items()}
num_classes = len(classes)
tokenizer = CharTokenizer()
tokenizer.fit(X_train)

maxlen = None
hdims = 256
vocab_size = tokenizer.vocab_size

inputs = Input(shape=(maxlen,))
x = Embedding(input_dim=vocab_size, output_dim=hdims, mask_zero=True)(inputs)
x = LayerNormalization()(x)
x = Bidirectional(LSTM(hdims, return_sequences=True), merge_mode="concat")(x)
x = Dense(hdims)(x)
x = Dropout(0.2)(x)
# x = LayerNormalization()(x)
x = Dense(num_classes)(x)
crf = CRF(trans_initializer="glorot_normal")
outputs = crf(x)

base = Model(inputs=inputs, outputs=outputs)
model = ModelWithCRFLoss(base)
model.summary()
model.compile(optimizer="adam")

batch_size = 32
epochs = 7
file = "weights/weights.task.ner.bilstm.crf"
ner = NamedEntityRecognizer(model, tokenizer, maxlen, id2label)
data_train = (X_train, y_train)
X_val, y_val = load_dataset("dev")
data_val = (X_val, y_val)
gen = batch_paded_generator(X_train, y_train, label2id, tokenizer, batch_size, epochs)
steps_per_epoch = len(X_train) // batch_size + 1
model.fit(
    gen,
    batch_size=batch_size,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    callbacks=[Evaluator(ner, data_train, "train"), Evaluator(ner, data_val, "dev")],
)

model.save_weights(file)


if __name__ == "__main__":
    X_test, y_test = load_dataset("test")
    # evaluate_prf(ner, X_train, y_train)
    evaluate_prf(ner, X_test, y_test)
    for x, y in zip(X_test, y_test):
        print(find_entities(x, y)) # 真实的实体
        print(ner.find(x)) # 预测的实体
        input()

# 输出结果示例
# [('基里延科', 'PER'), ('杜马', 'ORG'), ('叶利钦', 'PER')]
# [('基里延科', 'PER'), ('杜马', 'ORG'), ('叶利钦', 'PER')]
# [('美国', 'LOC'), ('克林顿', 'PER'), ('美国之音', 'ORG'), ('伊', 'LOC'), ('美国', 'LOC'), ('伊朗', 'LOC')]
# [('美国', 'LOC'), ('克林顿', 'PER'), ('美国', 'LOC'), ('伊', 'LOC'), ('美国', 'LOC'), ('伊朗', 'LOC')]

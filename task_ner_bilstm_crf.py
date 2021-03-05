import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import *

physical_devices = tf.config.experimental.list_physical_devices("GPU")
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import dataset
from crf import CRF, ModelWithCRFLoss
from evaluation import NEREvaluator, evaluate_prf
from snippets import CharTokenizer, NamedEntityRecognizer
from snippets import find_entities, batch_paded_generator
from snippets import plot_trans, compute_trans

load_dataset = dataset.load_msra
X_train, y_train, classes = load_dataset("train", with_labels=True)
id2label = {i:j for i,j in enumerate(classes)}
label2id = {j:i for i,j in id2label.items()}
num_classes = len(classes)
tokenizer = CharTokenizer()
tokenizer.fit(X_train)

maxlen = None
hdims = 128
vocab_size = tokenizer.vocab_size

inputs = Input(shape=(maxlen,))
x = Embedding(input_dim=vocab_size, output_dim=hdims, mask_zero=True)(inputs)
x = LayerNormalization()(x)
x = Bidirectional(LSTM(hdims, return_sequences=True), merge_mode="concat")(x)
x = Dense(hdims)(x)
x = Dense(num_classes)(x)
x = Dropout(0.3)(x)
crf = CRF(
    lr_multiplier=10,
    trans_initializer="glorot_normal",
    trainable=True
)
outputs = crf(x)

base = Model(inputs=inputs, outputs=outputs)
model = ModelWithCRFLoss(base)
model.summary()
model.compile(optimizer="adam")

batch_size = 32
epochs = 7
steps_per_epoch = len(X_train) // batch_size + 1
gen = batch_paded_generator(X_train, y_train, label2id, tokenizer, batch_size, epochs)

X_val, y_val = load_dataset("dev")
data_title_pairs = [(X_train, y_train, "train"), (X_val, y_val, "dev")]
ner = NamedEntityRecognizer(model, tokenizer, maxlen, id2label)
evaluator = NEREvaluator(ner, data_title_pairs)

if __name__ == "__main__":
    model.fit(
        gen,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[evaluator],
    )

    file = "weights/weights.task.ner.bilstm.crf"
    model.save_weights(file)
    trans = np.array(crf.trans)
    plot_trans(trans, classes)

    X_test, y_test = load_dataset("test")
    evaluate_prf(ner, X_test, y_test)
    for x, y in zip(X_test, y_test):
        print("true:", find_entities(x, y)) # 真实的实体
        print("pred:", ner.find(x)) # 预测的实体
        input()

# 输出结果示例
# [('基里延科', 'PER'), ('杜马', 'ORG'), ('叶利钦', 'PER')]
# [('基里延科', 'PER'), ('杜马', 'ORG'), ('叶利钦', 'PER')]
# [('美国', 'LOC'), ('克林顿', 'PER'), ('美国之音', 'ORG'), ('伊', 'LOC'), ('美国', 'LOC'), ('伊朗', 'LOC')]
# [('美国', 'LOC'), ('克林顿', 'PER'), ('美国', 'LOC'), ('伊', 'LOC'), ('美国', 'LOC'), ('伊朗', 'LOC')]

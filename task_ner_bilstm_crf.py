import random
import collections
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import *

# physical_devices = tf.config.experimental.list_physical_devices("GPU")
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

import dataset
from layers import MaskBiLSTM
from crf import CRF, ModelWithCRFLoss
from labels import gen_ner_labels
from snippets import *

# 标签映射
# NER标注一般使用IOBES或BIO，这里使用后者
labels, id2label, label2id = gen_ner_labels(["B", "I"], ["PER", "LOC", "ORG"])
num_classes = len(labels)

PATH = "dataset/china-people-daily-ner-corpus/example."
def load_dataset(file, shuffle=True):
    # 返回逐位置标注形式
    file = PATH + file
    with open(file, encoding="utf-8") as fp:
        text = fp.read()
    lines = text.split("\n\n")
    if shuffle:
        random.shuffle(lines)
    X = []
    y = []
    for line in lines:
        if not line:
            continue
        chars = []
        tags = []
        for item in line.split("\n"):
            char, label = item.split(" ")
            chars.append(char)
            tags.append(label2id[label])
        X.append("".join(chars))
        y.append(tags)

        assert len(chars) == len(tags)
    return X, y

# load_dataset = dataset.load_msra

X_train, y_train = load_dataset("train")
tokenizer = CharTokenizer()
tokenizer.fit(X_train)

maxlen = 128
hdims = 128
vocab_size = tokenizer.vocab_size

inputs = Input(shape=(maxlen,))
mask = Lambda(lambda x: tf.not_equal(x, 0))(inputs) # 全局mask
x = Embedding(input_dim=vocab_size, output_dim=hdims)(inputs)
x = MaskBiLSTM(hdims)(x, mask=mask)
x = Dense(hdims)(x)
x = Dense(num_classes)(x)
crf = CRF(trans_initializer="orthogonal")
# CRF需要mask来完成不定长序列的处理，这里是手动传入
# 可以设置Embedding参数mask_zero=True，避免手动传入
outputs = crf(x, mask=mask)

base = Model(inputs=inputs, outputs=outputs)
model = ModelWithCRFLoss(base)
model.summary()
model.compile(optimizer="adam")

X_train, y_train = preprocess_dataset(X_train, y_train, maxlen, tokenizer)
X_val, y_val = load_dataset("dev")
X_val, y_val = preprocess_dataset(X_val, y_val, maxlen, tokenizer)

batch_size = 32
epochs = 10
file = "weights/weights.task.ner.bilstm.crf"
model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_val, y_val)
)

model.save_weights(file)

if __name__ == "__main__":
    X_test, y_test = load_dataset("test")
    ner = NamedEntityRecognizer(model, tokenizer, maxlen, id2label)
    for x, y in zip(X_test, y_test):
        print(find_entities(x, y, id2label)) # 真实的实体
        print(ner.find(x)) # 预测的实体
        input()

# 输出结果示例
# [('基里延科', 'PER'), ('杜马', 'ORG'), ('叶利钦', 'PER')]
# [('基里延科', 'PER'), ('杜马', 'ORG'), ('叶利钦', 'PER')]
# [('美国', 'LOC'), ('克林顿', 'PER'), ('美国之音', 'ORG'), ('伊', 'LOC'), ('美国', 'LOC'), ('伊朗', 'LOC')]
# [('美国', 'LOC'), ('克林顿', 'PER'), ('美国', 'LOC'), ('伊', 'LOC'), ('美国', 'LOC'), ('伊朗', 'LOC')]

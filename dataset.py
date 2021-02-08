import collections
import itertools
import random
import re
import numpy as np
import labels

# NER数据加载

def load_file(file, sep=" ", shuffle=True):
    # 返回逐位置标注形式
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
            char, label = item.split(sep)
            if label.startswith("M"):
                # M -> I
                label = "I" + label[1:]
            chars.append(char)
            tags.append(label)
        X.append("".join(chars))
        y.append(tags)
        assert len(chars) == len(tags)
    return X, y

def load_dh_msra(file="dataset/dh_msra.txt", shuffle=True):
    # for evaluatoin
    return load_file(file, "\t", shuffle)

PATH_CPD = "dataset/china-people-daily-ner-corpus/example.{}"
def load_china_people_daily(file, shuffle=True):
    assert file in ("train", "dev", "test")
    file = PATH_CPD.format(file)
    return load_file(file, " ", shuffle)

PATH_MSRA = "dataset/ner/msra/{}.ner"
def load_msra(file, shuffle=True):
    file = PATH_MSRA.format(file)
    return load_file(file, " ", shuffle)

PATH_WB = "dataset/ner/weibo/{}.all.bmes"
def load_weibo(file, shuffle=True):
    file = PATH_WB.format(file)
    return load_file(file, " ", shuffle)

PATH_ON = "dataset/ner/ontonote4/{}.char.bmes"
def load_ontonote4(file, shuffle=True):
    file = PATH_ON.format(file)
    with open(file, "r") as fp:
        text = fp.read()
    lines = text.splitlines()
    X = []
    y = []
    for line in lines:
        sentence, tags = line.split("\t")
        X.append(sentence.replace(" ", ""))
        y.append(labels.bmes2iobes(tags.split(" ")))
    return X, y

PATH_RM = "dataset/ner/resume/{}.char.bmes"
def load_resume(file, shuffle=True):
    file = PATH_RM.format(file)
    with open(file, "r") as fp:
        text = fp.read()
    lines = text.splitlines()
    X = []
    y = []
    for line in lines:
        sentence, tags = line.split("\t")
        X.append(sentence.replace(" ", ""))
        y.append(labels.bmes2iobes(tags.split(" ")))
    return X, y


def train_val_test_split(X, y, train_size=0.7, val_size=0.2, test_size=0.1):
    pass


class CharTokenizer:
    pass

class BatchPaddedDataGenerator:
    pass

class DataGenerator:

    def __init__(self, sentences, char2id):
        self.sentences = sentences
        self.char2id = char2id

    def char_to_id(self, char):
        # 0 for MASK
        # 1 for UNK
        return self.char2id.get(char, 1)

    def __call__(self):
        for sentence in itertools.cycle(self.sentences):
            x = []
            y = []
            for word in sentence:
                for char in word:
                    x.append(self.char_to_id(char))
                if len(word) == 1:
                    # 's'
                    y.append(0)
                else:
                    # 'bme' or 'be'
                    y.extend([1] + [2]*(len(word)-2) + [3])

            x = np.array(x)
            # y = keras.utils.to_categorical(y, 4)
            yield (x, y)

# gen = DataGenerator(sentences, char2id)
# dataset = tf.data.Dataset.from_generator(
#     gen, (tf.int32, tf.float32)
# )

# dataset = dataset.prefetch(256)
# dataset = dataset.padded_batch(
#     batch_size=32, 
#     padded_shapes=([None], [None]), # [None, 4]
#     padding_values=(0, 0.0)
# )

# vocab_size = len(chars) + 2


def find_entities(text, tags):
    # 根据BIO标签提取文本中的实体
    def segment_by_tags(text, tags):
        buf = ""
        plabel = None
        for tag, char in zip(tags, text):
            tag = id2label[tag]
            if tag == "O":
                continue
            tag, label = tag.split("-")
            if tag == "B":
                if buf:
                    yield buf, plabel
                buf = char
            elif tag == "I":
                buf += char
            plabel = label

        if buf:
            yield buf, plabel
    return list(segment_by_tags(text, tags))


if __name__ == "__main__":
    load_dh_msra()
    for i in ("train", "dev", "test"):
        load_china_people_daily(i)
        load_msra(i)
        load_ontonote4(i)
        load_weibo(i)
        load_resume(i)

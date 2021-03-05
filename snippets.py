import collections
import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from labels import batch_tags2ids, ids2tags, find_entities

def pad(x, maxlen):
    x = sequence.pad_sequences(
        x, 
        maxlen=maxlen,
        dtype="int32",
        padding="post",
        truncating="post",
        value=0
    )
    return x

def batch_pad(x):
    maxlen = max([len(i) for i in x])
    return pad(x, maxlen)

def preprocess_dataset(X, y, maxlen, label2id, tokenizer):
    # 转成id序列并截断
    X = tokenizer.transform(X)
    y = batch_tags2ids(y, label2id)
    X = pad(X, maxlen)
    y = pad(y, maxlen)
    return X, y

def batch_paded_generator(X, y, label2id, tokenizer, batch_size, epochs):
    X = tokenizer.transform(X)
    y = batch_tags2ids(y, label2id)
    batchs = (len(X) // batch_size + 1) * epochs * batch_size
    X = itertools.cycle(X)
    y = itertools.cycle(y)
    gen = zip(X, y)
    batch_X = []
    batch_y = []
    for _ in range(batchs):
        sample_x, sample_y = next(gen)
        batch_X.append(sample_x)
        batch_y.append(sample_y)
        if len(batch_X) == batch_size:
            yield batch_pad(batch_X), batch_pad(batch_y)
            batch_X = []
            batch_y = []

def compute_trans(seqs, states):
    states2id = {i:j for j,i in enumerate(states)}
    size = len(states)
    trans = np.zeros((size, size))
    for seq in seqs:
        for state1, state2 in zip(seq[:-1], seq[1:]):
            id1 = states2id[state1]
            id2 = states2id[state2]
            trans[id1][id2] += 1
    trans = trans / np.sum(trans, axis=1, keepdims=True)
    return trans

def plot_trans(A, tags, show=True):
    # 可视化状态矩阵A
    ax = sns.heatmap(
        A,
        vmin=0.0,
        vmax=1.0,
        fmt=".2f",
        cmap="copper",
        annot=True,
        cbar=True,
        linewidths=0.25,
        cbar_kws={"orientation": "horizontal"}
    )
    ax.set_title("Transition Matrix")
    ax.set_xticklabels(tags, rotation=0)
    ax.set_yticklabels(tags, rotation=0)
    if show:
        plt.show()

class CharTokenizer:
    """字符级别Tokenizer"""

    char2id = {}
    MASK = 0
    UNK = 1 # 未登陆字用1表示

    def fit(self, X):
        chars = collections.defaultdict(int)
        for sample in X:
            for c in sample:
                chars[c] += 1
        self.char2id = {j:i for i,j in enumerate(chars, start=2)}
        self.id2char = {j:i for i,j in self.char2id.items()}

    def transform(self, X):
        # 转成ID序列
        ids = []
        for sentence in X:
            s = []
            for char in sentence:
                s.append(self.char2id.get(char, self.UNK))
            ids.append(s)
        return ids

    def inverse_transform(self, batch_ids):
        X = []
        for ids in batch_ids:
            x = []
            for i in ids:
                x.append(self.id2char.get(i))
            X.append(x)
        return X

    @property
    def vocab_size(self):
        return len(self.char2id) + 2

class LabelTransformer:

    def fit(self, batch_y):
        pass

    def transform(self, batch_ids):
        pass

# def find_entities(text, tags):
#     # 根据标签提取文本中的实体
#     # 适合BIO和BIOES标签
#     def segment_by_tags(text, tags):
#         buf = ""
#         plabel = None
#         for tag, char in zip(tags, text):
#             if tag == "O":
#                 continue
#             tag, label = tag.split("-")
#             if tag == "B" or tag == "S":
#                 if buf:
#                     yield buf, plabel
#                 buf = char
#             elif tag == "I" or tag == "E":
#                 buf += char
#             plabel = label

#         if buf:
#             yield buf, plabel
#     return list(segment_by_tags(text, tags))

def viterbi_decode(scores, trans, return_score=False):
    # 使用viterbi算法求最优路径
    # scores.shape = (seq_len, num_tags)
    # trans.shape = (num_tags, num_tags)
    dp = np.zeros_like(scores)
    backpointers = np.zeros_like(scores, dtype=np.int32)
    dp[0] = scores[0]
    for t in range(1, scores.shape[0]):
        v = np.expand_dims(dp[t-1], axis=1) + trans
        dp[t] = scores[t] + np.max(v, axis=0)
        backpointers[t] = np.argmax(v, axis=0)

    viterbi = [np.argmax(dp[-1])]
    for bp in reversed(backpointers[1:]):
        viterbi.append(bp[viterbi[-1]])
    viterbi.reverse()
    if return_score:
        viterbi_score = np.max(dp[-1])
        return viterbi, viterbi_score
    return viterbi

class NamedEntityRecognizer:
    """封装好的实体识别器"""

    def __init__(self, model, tokenizer, maxlen, id2label):
        self.model = model
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.id2label = id2label

    def find(self, text):
        size = len(text)
        ids = self.tokenizer.transform([text])
        padded_ids = pad(ids, size)
        tags = self.model.predict(padded_ids)[0]
        tags = ids2tags(tags[:size], self.id2label)
        return find_entities(text, tags)

    def batch_find(self, texts):
        lengths = [len(text) for text in texts]
        ids = self.tokenizer.transform(texts)
        padded_ids = batch_pad(ids)
        batch_tags = self.model.predict(padded_ids)
        batch_entities = []
        for size, tags, text in zip(lengths, batch_tags, texts):
            tags = ids2tags(tags[:size], self.id2label)
            batch_entities.append(find_entities(text, tags))
        return batch_entities

class ViterbiNamedEntityRecognizer:
    """带Viterbi解码的实体识别器"""

    def __init__(self, model, trans, tokenizer, maxlen, id2label, method="viterbi"):
        self.model = model
        self.trans = trans
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.id2label = id2label
        assert method in ("viterbi", "greedy")
        self.method = method

    def decode(self, scores):
        if self.method == "viterbi":
            return viterbi_decode(scores, self.trans)
        else:
            return np.argmax(scores, axis=1).tolist()

    def find(self, text):
        size = len(text)
        ids = self.tokenizer.transform([text])
        padded_ids = pad(ids, self.maxlen)
        scores = self.model.predict(padded_ids)[0][:size]
        tags = self.decode(scores)
        tags = ids2tags(tags, self.id2label)
        return find_entities(text, tags)

    def batch_find(self, texts):
        lengths = [len(text) for text in texts]
        ids = self.tokenizer.transform(texts)
        padded_ids = batch_pad(ids)
        batch_scores = self.model.predict(padded_ids)
        batch_entities = []
        for size, scores, text in zip(lengths, batch_scores, texts):
            tags = self.decode(scores[:size])
            tags = ids2tags(tags, self.id2label)
            batch_entities.append(find_entities(text, tags))
        return batch_entities

class TransitionMatrixInitializer(tf.keras.initializers.Initializer):
    """状态矩阵的初始化"""

    def __init__(self, trans):
        self.trans = trans

    def __call__(self, shape, dtype=None):
        return self.trans

    def get_config(self):
        return {"trans": self.trans}

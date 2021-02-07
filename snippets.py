import collections
from tensorflow.keras.preprocessing import sequence

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

def preprocess_dataset(X, y, maxlen, tokenizer):
    # 转成id序列并截断
    X = tokenizer.transform(X)
    X = pad(X, maxlen)
    y = pad(y, maxlen)
    return X, y

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

    def transform(self, X):
        # 转成ID序列
        ids = []
        for sentence in X:
            s = []
            for char in sentence:
                s.append(self.char2id.get(char, self.UNK))
            ids.append(s)
        return ids

    @property
    def vocab_size(self):
        return len(self.char2id) + 2

def find_entities(text, tags, id2label):
    # 根据标签提取文本中的实体
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
        padded_ids = pad(ids, self.maxlen)
        tags = self.model.predict(padded_ids)[0][0]
        tags = tags[:size]
        return find_entities(text, tags, self.id2label)

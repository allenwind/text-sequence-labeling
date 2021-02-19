
# === dataset ===
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

gen = DataGenerator(sentences, char2id)
dataset = tf.data.Dataset.from_generator(
    gen, (tf.int32, tf.float32)
)

dataset = dataset.prefetch(256)
dataset = dataset.padded_batch(
    batch_size=32, 
    padded_shapes=([None], [None]), # [None, 4]
    padding_values=(0, 0.0)
)

vocab_size = len(chars) + 2

# == labels.py ==
def split_chunks(text, tags, maxlen):
    # 把长样本切分多个块
    if len(text) <= maxlen:
        return [text], [tags]

def merge_chunks(chunks):
    # 合并多个chunks
    pass

def find_entities_by_bio_tags(text, tags):
    # 根据BIO标签提取文本中的实体
    def segment_by_tags(text, tags):
        buf = ""
        plabel = None
        for tag, char in zip(tags, text):
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

def find_entities_by_iobes_tags(text, tags):
    # 根据IOBES标签提取文本中的实体
    def segment_by_tags(text, tags):
        buf = ""
        plabel = None
        for tag, char in zip(tags, text):
            if tag == "O":
                continue
            tag, label = tag.split("-")
            if tag == "B" or tag == "S":
                if buf:
                    yield buf, plabel
                buf = char
            elif tag == "I" or tag == "E":
                buf += char
            plabel = label

        if buf:
            yield buf, plabel
    return list(segment_by_tags(text, tags))

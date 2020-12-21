import collections
import itertools
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras

# 中文分词数据生成

_PKU = ""
def load_icwb2_pku(file=_PKU):
	return X, y, cates

def load_icb2_msr(file=_MSR):
    pass

def load_cpd_ner(file=_CPD):
    # china-people-daily-ner-corpus
    pass

file = "/home/zhiwen/workspace/dataset/icwb2-data/training/msr_training.utf8"
with open(file, encoding="utf-8") as fp:
	text = fp.read()

sentences = text.splitlines()
sentences = [re.split(" +", s) for s in sentences]
sentences = [[word for word in sentence if word] for sentence in sentences]

chars = collections.defaultdict(int)
for sentence in sentences:
	for word in sentence:
		for char in word:
			chars[char] += 1

sentences = [sentence for sentence in sentences if len(sentence) > 1]

min_freq = 1
chars = {i:j for i, j in chars.items() if j >= min_freq}
id2char = {i:j for i, j in enumerate(chars, start=1)}
char2id = {j:i for i, j in id2char.items()}

id2tag = {0:'s', 1:'b', 2:'m', 3:'e'}
tag2id = {j:i for i, j in id2tag.items()}

class Tokenizer:
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


import tensorflow as tf

# 实体识别数据生成

path = "/home/zhiwen/workspace/dataset/ner/china-people-daily-ner-corpus/example.train"

# def load_data(filename):
#     D = []
#     with open(filename, encoding='utf-8') as f:
#         f = f.read()
#         for l in f.split('\n\n'):
#             if not l:
#                 continue
#             d, last_flag = [], ''
#             for c in l.split('\n'):
#                 char, this_flag = c.split(' ')
#                 if this_flag == 'O' and last_flag == 'O':
#                     d[-1][0] += char
#                 elif this_flag == 'O' and last_flag != 'O':
#                     d.append([char, 'O'])
#                 elif this_flag[:1] == 'B':
#                     d.append([char, this_flag[2:]])
#                 else:
#                     d[-1][0] += char
#                 last_flag = this_flag
#             D.append(d)
#     return D

def load_data(filename):
    with open(filename, encoding="utf-8") as fd:
        text = fd.read()
    lines = [line for line in text.split("\n\n") if line]
    for line in lines:
        X = []
        y = []
        for char_label in line.split("\n"):
            char, label = char_label.split(" ")
            X.append(char)
            y.append(label)
        yield X, y

gen = load_data(path)
X, y = next(gen)
print(X)
print(y)
X, y = next(gen)
print(X)
print(y)



if __name__ == "__main__":
    import time
    c = time.time()
    for x, y in iter(dataset):
        print(x.shape, y.shape)
    print(time.time()-c)

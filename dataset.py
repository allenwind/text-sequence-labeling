import collections
import itertools
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras

# 中文分词数据生成

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

if __name__ == "__main__":
    import time
    c = time.time()
    for x, y in iter(dataset):
        print(x.shape, y.shape)
    print(time.time()-c)

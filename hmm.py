import itertools
import random

import numpy as np
import matplotlib.pyplot as plt

class HMM:
	pass


def compute_state_matrix(ss):
	states = {j: i for i, j in enumerate(set(itertools.chain(*ss)))}
	size = len(states)
	matrix = np.zeros((size, size))
	for s in ss:
		for i in range(len(s)-1):
			a = states[s[i]]
			b = states[s[i+1]]
			matrix[a, b] += 1
	return matrix / np.sum(matrix, axis=0)


def gen_io(A, B, pi, I, O, T=100):
	i = ""
	o = ""
	for t in range(T):
		j = np.argmax(pi)
		i += I[j]
		k = np.argmax(B[:, j])
		o += O[k]
		# 状态转移
		pi = np.dot(pi, A)
	return (i, o)

def compute_lambda(ios):
	# 参数估计
	pass


def gen_ss():
	s = "abcdd" * 5
	ss = []
	for _ in range(100):
		ss.append("".join(random.sample(s, 8)))
	return ss

ss = gen_ss()
m = compute_state_matrix(ss)
print(m)
plt.imshow(m)
plt.show()


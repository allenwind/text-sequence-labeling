import itertools
import random
import operator

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

def viterbi(nodes, weight)
    paths = nodes[0] # 初始化起始路径
    for l in range(1, len(nodes)): # 遍历后面的节点
        paths_old, paths = paths, {}
        for n, ns in nodes[l].items(): # 当前时刻的所有节点
            max_path, max_score = '', -1e10
            for p, ps in paths_old.items(): # 截止至前一时刻的最优路径集合
                score = ns + ps + weight[p[-1]+n] # 计算新分数
                if score > max_score: # 如果新分数大于已有的最大分
                    max_path, max_score = p + n, score # 更新路径
            paths[max_path] = max_score # 储存到当前时刻所有节点的最优路径
    return max(path.items(), key=operator.itemgetter(1))

import itertools
import random
import math
import numpy as np
from collections import *
from labels import find_entities

class HiddenMarkovChain:
    
    def __init__(self, tags):
        # 标签集
        self.tags = sorted(set(tags))
        self.tags2id = {i:j for j,i in enumerate(self.tags)}
        self.id2tags = {j:i for i,j in self.tags2id.items()}
        self.state_size = len(tags)
        self.convert = find_entities
        self.reset()

    def reset(self):
        # 状态转移矩阵
        self.A = np.zeros((self.state_size, self.state_size))
        # 观察矩阵，稀疏形式
        self.B = defaultdict(Counter)
        self.built = False

    def fit(self, X, y):
        if self.built:
            self.reset()
        # 状态转移矩阵参数学习
        for labels in y:
            for label1, label2 in zip(labels[:-1], labels[1:]):
                id1 = self.tags2id[label1]
                id2 = self.tags2id[label2]
                self.A[id1][id2] += 1
        self.A = self.A / np.sum(self.A, axis=1, keepdims=True)
        # 观察矩阵参数学习
        for sentence, labels in zip(X, y):
            for char, label in zip(sentence, labels):
                self.B[label][char] += 1
        self.logtotal = {tag:math.log(sum(self.B[tag].values())) for tag in self.tags}
        self.built = True

    def predict(self, X):
        batch_scores = []
        for sentence in X:
            scores = np.zeros((len(sentence), self.state_size))
            for i, char in enumerate(sentence):
                for j, k in self.B.items():
                    scores[i][self.tags2id[j]] = math.log(k[char]+1) - self.logtotal[j]
            batch_scores.append(scores)
        return batch_scores

    def find(self, sentence):
        scores = self.predict([sentence])[0]
        log_trans = np.log(np.where(self.A==0, 0.0001, self.A))
        viterbi = self.viterbi_decode(scores, log_trans)
        viterbi = [self.id2tags[i] for i in viterbi]
        return self.convert(sentence, viterbi)

    def viterbi_decode(self, scores, trans, return_score=False):
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

    def plot_trans(self):
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
        except ImportError as err:
            print(err)
            return
        ax = sns.heatmap(
            self.A,
            vmin=0.0,
            vmax=1.0,
            fmt=".2f",
            cmap="copper",
            annot=True,
            cbar=True,
            # xticklabels=self.tags,
            # yticklabels=self.tags,
            linewidths=0.25,
            cbar_kws={"orientation": "horizontal"}
        )
        ax.set_title("Transition Matrix")
        ax.set_xticklabels(self.tags, rotation=0)
        ax.set_yticklabels(self.tags, rotation=0)
        plt.show()

if __name__ == "__main__":
    import dataset
    load_dataset = dataset.load_msra
    X, y, labels = load_dataset("train", with_labels=True)
    model = HiddenMarkovChain(labels)
    model.fit(X, y)
    model.plot_trans()

    X, y = load_dataset("test")
    for sentence, labels in zip(X, y):
        print(model.find(sentence))
        print(find_entities(sentence, labels))
        input()

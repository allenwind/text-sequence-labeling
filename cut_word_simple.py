import glob
import math
import collections
import ahocorasick

path = '/home/zhiwen/workspace/dataset/THUOCL中文分类词库'
def load_dict(path=path, proba=True):
    files = glob.glob(path + "/*.txt")
    wf_dict = collections.defaultdict(int)
    for file in files:
        with open(file, encoding="utf-8") as fd:
            lines = fd.readlines()
        for line in lines:
            try:
                word, freq = line.strip().split("\t")
                wf_dict[word] += int(freq)
            except ValueError:
                print(line, file)

    if proba:
        total = sum(wf_dict.values())
        wf_dict = {i:j/total for i, j in wf_dict.items()}
    wf_dict = {i:j for i, j in wf_dict.items() if j > 0}
    return wf_dict

wf_dict = load_dict()
# for i, j in wf_dict.items():
#     print(i, j)

am = ahocorasick.Automaton()
for word, proba in wf_dict.items():
    am.add_word(word, (word, math.log(proba)))

am.make_automaton()

sentence = "人的复杂的生理系统的特性注定了一件事情，就是从懂得某个道理到执行之间，是一个漫长的回路。"
for i, j in am.iter(sentence):
    print(j[0], end=' ')

import collections
import itertools
import random
import re
import numpy as np
from labels import bmes2iobes, iobes2bio

# NER数据加载

def load_recursion():
    # TODO
    if file not in ("train", "dev", "test"):
        if file == "all":
            files = ("train", "dev", "test")
        else:
            files = file.split("+")
        X = []
        y = []
        for file in files:
            r = load_file(file)
            X.extend(r[0])
            y.extend(r[1])
        if with_labels:
            labels = set(itertools.chain(*y))
            return X, y, sorted(labels)
        return X, y

def load_file(file, sep=" ", shuffle=True, with_labels=False):
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
        y.append(iobes2bio(tags))
        assert len(chars) == len(tags)
    if with_labels:
        labels = set(itertools.chain(*y))
        return X, y, sorted(labels)
    return X, y

def load_dh_msra(file="dataset/dh_msra.txt", shuffle=True, with_labels=False):
    # for evaluatoin
    return load_file(file, "\t", shuffle, with_labels)

PATH_CPD = "dataset/ner/china-people-daily-ner-corpus/example.{}"
def load_china_people_daily(file, shuffle=True, with_labels=False):
    file = PATH_CPD.format(file)
    return load_file(file, " ", shuffle, with_labels)

# china_people_daily 简称
load_cpd = load_china_people_daily

PATH_MSRA = "dataset/ner/msra/{}.ner"
def load_msra(file, shuffle=True, with_labels=False):
    file = PATH_MSRA.format(file)
    return load_file(file, " ", shuffle, with_labels)

PATH_WB = "dataset/ner/weibo/{}.all.bmes"
def load_weibo(file, shuffle=True, with_labels=False):
    file = PATH_WB.format(file)
    return load_file(file, " ", shuffle, with_labels)

PATH_ON = "dataset/ner/ontonote4/{}.char.bmes"
def load_ontonote4(file, shuffle=True, with_labels=False):
    file = PATH_ON.format(file)
    with open(file, "r") as fp:
        text = fp.read()
    lines = text.splitlines()
    X = []
    y = []
    for line in lines:
        sentence, tags = line.split("\t")
        X.append(sentence.replace(" ", ""))
        y.append(bmes2iobes(tags.split(" ")))
    if with_labels:
        labels = set(itertools.chain(*y))
        return X, y, sorted(labels)
    return X, y

PATH_RM = "dataset/ner/resume/{}.char.bmes"
def load_resume(file, shuffle=True, with_labels=False):
    file = PATH_RM.format(file)
    with open(file, "r") as fp:
        text = fp.read()
    lines = text.splitlines()
    X = []
    y = []
    for line in lines:
        sentence, tags = line.split("\t")
        X.append(sentence.replace(" ", ""))
        y.append(bmes2iobes(tags.split(" ")))

    if with_labels:
        labels = set(itertools.chain(*y))
        return X, y, sorted(labels)
    return X, y

def inspect_data(load):
    import matplotlib.pyplot as plt
    from snippets import plot_trans, compute_trans
    i = 1
    for ds in ("train", "dev", "test"):
        plt.subplot(1, 3, i)
        i += 1
        X, y, classes = load(ds, with_labels=True)
        trans = compute_trans(y, classes)
        plot_trans(trans, classes, show=False)
    plt.show()

if __name__ == "__main__":
    # for testing

    load_dh_msra()
    for i in ("train", "dev", "test"):
        X, y, classes = load_china_people_daily(i, with_labels=True)
        print(len(X))
        print(classes)
        X, y, classes = load_msra(i, with_labels=True)
        print(len(X))
        print(classes)
        X, y, classes = load_ontonote4(i, with_labels=True)
        print(len(X))
        print(classes)
        X, y, classes = load_weibo(i, with_labels=True)
        print(len(X))
        print(classes)
        X, y, classes = load_resume(i, with_labels=True)
        print(len(X))
        print(classes, end="\n\n")

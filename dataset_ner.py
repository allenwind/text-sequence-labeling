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

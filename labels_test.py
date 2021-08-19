import dataset
from labels import find_entities

# 测试标签解码的正确性

load_dataset = dataset.load_msra
X, y = load_dataset("train", True)

def find_entities2(text, tags, withO=False):
    d, last_flag = [], ''
    for char, this_flag in zip(text, tags):
        if this_flag == 'O' and last_flag == 'O':
            d[-1][0] += char
        elif this_flag == 'O' and last_flag != 'O':
            d.append([char, 'O'])
        elif this_flag[:1] == 'B':
            d.append([char, this_flag[2:]])
        else:
            d[-1][0] += char
        last_flag = this_flag

    d = [(i,j) for i,j in d]
    if not withO:
        d = [(i,j) for i,j in d if j != "O"]
    return d

# 验证find_entities与find_entities2的兼容性
def check():
    for text, tags in zip(X, y):
        r1 = find_entities(text, tags, True)
        r2 = find_entities2(text, tags, True)
        for i, j in zip(r1, r2):
            if i != j:
                print(i)
                print(j)
                raise

if __name__ == "__main__":
    print("check...")
    check()
    print("check pass")

    for text, tags in zip(X, y):
        rs = find_entities(text, tags, withO=True)
        print(rs)
        print("|".join(tags))

        ctext = "".join([i[0] for i in rs])
        assert text == ctext
        print(text)
        print(ctext)

        input()

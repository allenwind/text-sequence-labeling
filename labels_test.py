import dataset

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

def find_entities(text, tags, withO=False):
    # 根据标签提取文本中的实体
    # 适合BIO和BIOES标签
    # withO是否返回O标签内容
    def segment_by_tags(text, tags):
        buf = ""
        for tag, char in zip(tags, text):
            if tag == "O":
                label = tag
            else:
                tag, label = tag.split("-")
            if tag == "B" or tag == "S":
                if buf:
                    yield buf, plabel
                buf = char
                plabel = label
            elif tag == "I" or tag == "E":
                buf += char
            elif withO and tag == "O":
                # tag == "O"
                if buf and plabel != "O":
                    yield buf, plabel
                    buf = ""
                buf += char
                plabel = label
        if buf:
            yield buf, plabel
    return list(segment_by_tags(text, tags))

def check():
    for text, tags in zip(X, y):
        r1 = find_entities(text, tags, True)
        r2 = find_entities2(text, tags, True)
        for i, j in zip(r1, r2):
            if i != j:
                print(i)
                print(j)
                raise

print("check...")
# check()
print("check pass")

for text, tags in zip(X, y):
    rs = find_entities(text, tags, withO=True)
    print(rs)
    print("|".join(tags))

    ctext = "".join([i[0] for i in rs])
    print(text)
    print(ctext)

    # assert text == ctext

    input()

import itertools

# 标签相关处理函数

def gen_ner_labels(tags, clabels, withO=True):
    labels = itertools.product(tags, clabels)
    labels = ["-".join(i) for i in labels]
    if withO:
        labels.append("O")
    id2label = {i:j for i,j in enumerate(labels)}
    label2id = {j:i for i,j in id2label.items()}
    return labels, id2label, label2id

IOBES = list("SBIE")
BIO = list("BI")

def ids2tags(ids, id2label):
    # 标签ID序列转标签序列
    return [id2label[i] for i in ids]

def batch_ids2tags(batch_ids, id2label):
    pass

def tags2ids(tags, label2id):
    # 标签序列转标签ID序列
    return [label2id[tag] for tag in tags]

def batch_tags2ids(batch_tags, label2id):
    pass

def bio2iobes(tags):
    # BIO标签转IOBES标签
    def split_spans(tags):
        buf = []
        for tag in tags:
            if tag == "O" or tag.startswith("B"):
                if buf:
                    yield buf
                buf = [tag]
            else:
                # tag.startswith("I")
                buf.append(tag)
        if buf:
            yield buf

    ntags = []
    for span in split_spans(tags):
        tag = span[0]
        if len(span) == 1:
            if tag == "O":
                ntags.append(tag)
            else:
                tag = "S" + tag[1:]
                ntags.append(tag)
        else:
            btag = "B" + tag[1:]
            itag = "I" + tag[1:]
            etag = "E" + tag[1:]
            span_tags = [btag] + [itag] * (len(span) - 2) + [etag]
            ntags.extend(span_tags)
    return ntags

def iobes2bio(tags):
    # IOBES标签转BIO标签
    ntags = []
    for tag in tags:
        if tag == "O":
            ntags.append(tag)
            continue
        tag, label = tag.split("-")
        if tag == "E":
            tag = "I"
        if tag == "S":
            tag = "B"
        tag = tag + "-" + label
        ntags.append(tag)
    return ntags

def split_into_tags_labels(tags):
    # 实体标注和实体类别分离
    ntags = []
    labels = []
    for tag in tags:
        if tag == "O":
            ntags.append(tag)
            labels.append(tag)
            continue
        tag, label = tag.split("-")
        ntags.append(tag)
        labels.append(label)
    return ntags, labels

def find_entities_by_bio_tags(text, tags):
    # 根据BIO标签提取文本中的实体
    def segment_by_tags(text, tags):
        buf = ""
        plabel = None
        for tag, char in zip(tags, text):
            if tag == "O":
                continue
            tag, label = tag.split("-")
            if tag == "B":
                if buf:
                    yield buf, plabel
                buf = char
            elif tag == "I":
                buf += char
            plabel = label

        if buf:
            yield buf, plabel
    return list(segment_by_tags(text, tags))

def find_entities_by_iobes_tags(text, tags):
    # 根据IOBES标签提取文本中的实体
    def segment_by_tags(text, tags):
        buf = ""
        plabel = None
        for tag, char in zip(tags, text):
            if tag == "O":
                continue
            tag, label = tag.split("-")
            if tag == "B" or tag == "S":
                if buf:
                    yield buf, plabel
                buf = char
            elif tag == "I" or tag =="E":
                buf += char
            plabel = label

        if buf:
            yield buf, plabel
    return list(segment_by_tags(text, tags))

def split_chunks(text, tags, maxlen):
    # 把长样本切分多个块
    if len(text) <= maxlen:
        return [text], [tags]

def merge_chunks(chunks):
    # 合并多个chunks
    pass

if __name__ == "__main__":
    text = "AABCCCCCDD"
    tags = ["O", "O", "B-LOC", "B-LOC", "I-LOC", "I-LOC", "I-LOC", "I-LOC", "O", "O"]
    iobes_tags = bio2iobes(tags)
    bio_tags = iobes2bio(iobes_tags)
    print(tags)
    print(bio_tags)
    print(iobes_tags)

    ntags, labels = split_into_tags_labels(tags)
    print(ntags)
    print(labels)

    print(find_entities_by_bio_tags(text, bio_tags))
    print(find_entities_by_iobes_tags(text, iobes_tags))

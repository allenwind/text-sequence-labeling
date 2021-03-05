import itertools
import collections

# 标签相关处理函数
# BIOES/BIO/BMESO

def gen_ner_labels(tags, clabels, withO=True):
    labels = itertools.product(tags, clabels)
    labels = ["-".join(i) for i in labels]
    if withO:
        labels.append("O")
    id2label = {i:j for i,j in enumerate(labels)}
    label2id = {j:i for i,j in id2label.items()}
    return labels, id2label, label2id

BIO = ["B", "I"]
IOBES = ["S", "B", "I", "E"]
BMES = ["B", "M", "E", "S"]

def find_tag_type(batch_y, limit=None):
    # 确定标签类型
    s = set()
    for tags in batch_y[:limit]:
        for tag in tags:
            if tag == "O":
                continue
            tag, label = tag.split("-", 1)
            s.add(tag)
    if len(s) == 2:
        return BIO
    if "M" in s:
        return BMES
    return IOBES

def find_clabels(batch_y, limit=None):
    # 确定类别集合
    clabels = set()
    for tags in batch_y[:limit]:
        for tag in tags:
            if tag == "O":
                continue
            tag, label = tag.split("-", 1)
            clabels.add(label)
    return sorted(clabels)

def labels_counter(y):
    # 标签统计
    y = itertools.chain(*y)
    c = collections.Counter(y)
    return c

def ids2tags(ids, id2label):
    # 标签ID序列转标签序列
    return [id2label[i] for i in ids]

def batch_ids2tags(batch_ids, id2label):
    batch_tags = []
    for ids in batch_ids:
        batch_tags.append(ids2tags(ids, id2label))
    return batch_tags

def tags2ids(tags, label2id):
    # 标签序列转标签ID序列
    return [label2id[tag] for tag in tags]

def batch_tags2ids(batch_tags, label2id):
    batch_ids = []
    for tags in batch_tags:
        batch_ids.append(tags2ids(tags, label2id))
    return batch_ids

class TaggingTransformer:
    """标签映射，标签的转换和逆转换"""

    def fit(self, batch_tags):
        self.labels = set(itertools.chain(*batch_tags))
        self.id2label = {i:j for i,j in enumerate(self.labels)}
        self.label2id = {j:i for i,j in self.id2label.items()}

    def transform(self, batch_tags):
        batch_ids = []
        for tags in batch_tags:
            ids = []
            for tag in tags:
                ids.append(self.label2id[tag])
            batch_ids.append(ids)
        return batch_ids

    def inverse_transform(self, batch_ids):
        batch_tags = []
        for ids in batch_ids:
            tags = []
            for i in ids:
                tags.append(self.id2label[i])
            batch_tags.append(tags)
        return batch_tags

    @property
    def num_classes(self):
        return len(self.labels)

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

def bmes2iobes(tags):
    # BMES标签转成IOBES标签
    # 实则就是 M -> I
    ntags = []
    for tag in tags:
        if tag.startswith("M"):
            tag = "I" + tag[1:]
        ntags.append(tag)
    return ntags

def bmes2bio(tags):
    iobes_tags = bmes2iobes(tags)
    bio_tags = iobes2bio(iobes_tags)
    return bio_tags

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

def find_entities(text, tags, withO=False):
    # 根据标签提取文本中的实体
    # 适合BIO和BIOES标签
    # withO是否返回O标签内容
    def segment_by_tags(text, tags):
        buf = ""
        plabel = None
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
                if buf and plabel != "O":
                    yield buf, plabel
                    buf = ""
                buf += char
                plabel = label
        if buf:
            yield buf, plabel
    return list(segment_by_tags(text, tags))

if __name__ == "__main__":
    import dataset

    text = "AABCCCCCDD"
    tags = ["O", "O", "B-LOC", "B-LOC", "I-LOC", "I-LOC", "I-LOC", "I-LOC", "O", "O"]
    
    print(find_tag_type([tags]))
    print(find_clabels([tags]))

    labels, id2label, label2id = gen_ner_labels(IOBES, ["LOC", "PER", "ORG"])
    print(labels)
    print(id2label)
    print(label2id)

    iobes_tags = bio2iobes(tags)
    bio_tags = iobes2bio(iobes_tags)
    print(tags)
    print(bio_tags)
    print(iobes_tags)

    ntags, labels = split_into_tags_labels(tags)
    print(ntags)
    print(labels)

    print(find_entities(text, bio_tags, withO=True))
    print(find_entities(text, iobes_tags, withO=True))

    # check
    for ds in ("train", "dev", "test"):
        X, y = dataset.load_msra(ds)
        for text, tags in zip(X, y):
            es = find_entities(text, tags, withO=True)
            assert "".join([i[0] for i in es]) == text

import tensorflow as tf
import tqdm
from snippets import find_entities

class NEREvaluator(tf.keras.callbacks.Callback):
    """prf指标评估"""

    def __init__(self, ner, data_title_pairs):
        self.ner = ner
        # (X, y, title)
        self.data_title_pairs = data_title_pairs

    def evaluate_prf(self, X, y):
        Rs = self.ner.batch_find(X) # 预测正类
        Ts = [find_entities(text, tags) for text, tags in zip(X, y)] # 真正类
        TP = TPFP = TPFN = 1e-12
        for R, T in zip(Rs, Ts):
            R = set(R)
            T = set(T)
            TP += len(R & T) # TP
            TPFP += len(R) # TP + FP
            TPFN += len(T) # TP + FN
        p = TP / TPFP * 100
        r = TP / TPFN * 100
        f1 = 2 * TP / (TPFP + TPFN) * 100
        return p, r, f1

    def on_epoch_end(self, epoch, logs=None):
        print()
        for X, y, title in self.data_title_pairs:        
            p, r, f1 = self.evaluate_prf(X, y)
            template = '{} - precision: {:.2f}% - recall: {:.2f}% - f1: {:.2f}%'
            print(template.format(title, p, r, f1))

class CWSEvaluator(tf.keras.callbacks.Callback):
    pass

def evaluate_prf(ner, X, y):
    trues = [find_entities(text, tags) for text, tags in zip(X, y)] # 真正类
    preds = ner.batch_find(X) # 预测正类
    TP = TPFP = TPFN = 1e-12
    for R, T in zip(preds, trues):
        TP += len(set(R) & set(T)) # TP
        TPFP += len(R) # TP + FP
        TPFN += len(T) # TP + FN
    p = TP / TPFP * 100.0
    r = TP / TPFN * 100.0
    f1 = 2 * p * r / (p + r)
    template = 'precision: {:.2f}% - recall: {:.2f}% - f1: {:.2f}%'
    print(template.format(p, r, f1))
    return p, r, f1

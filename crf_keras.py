# -*- coding:utf-8 -*-

import tensorflow as tf


class CRF(tf.keras.layers.Layer):
    """纯Keras实现CRF层
    CRF层本质上是一个带训练参数的loss计算层，因此CRF层只用来训练模型，
    而预测则需要另外建立模型。
    """

    def __init__(self, ignore_last_label=False, **kwargs):
        """ignore_last_label：定义要不要忽略最后一个标签，起到mask的效果
        """
        self.ignore_last_label = 1 if ignore_last_label else 0
        super(CRF, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num_labels = input_shape[-1] - self.ignore_last_label
        self.trans = self.add_weight(name='crf_trans',
                                     shape=(self.num_labels, self.num_labels),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def log_norm_step(self, inputs, states):
        """递归计算归一化因子
        要点：1、递归计算；2、用logsumexp避免溢出。
        技巧：通过expand_dims来对齐张量。
        """
        states = tf.expand_dims(states[0], 2)  # (batch_size, output_dim, 1)
        trans = tf.expand_dims(self.trans, 0)  # (1, output_dim, output_dim)
        output = tf.reduce_logsumexp(states+trans, 1)  # (batch_size, output_dim)
        return output+inputs, [output+inputs]

    def path_score(self, inputs, labels):
        """计算目标路径的相对概率（还没有归一化）
        要点：逐标签得分，加上转移概率得分。
        技巧：用“预测”点乘“目标”的方法抽取出目标路径的得分。
        """
        point_score = tf.reduce_sum(tf.reduce_sum(inputs*labels, 2), 1, keepdims=True)  # 逐标签得分
        labels1 = tf.expand_dims(labels[:, :-1], 3)
        labels2 = tf.expand_dims(labels[:, 1:], 2)
        labels = labels1 * labels2  # 两个错位labels，负责从转移矩阵中抽取目标转移得分
        trans = tf.expand_dims(tf.expand_dims(self.trans, 0), 0)
        trans_score = tf.reduce_sum(tf.reduce_sum(trans*labels, [2, 3]), 1, keepdims=True)
        return point_score+trans_score  # 两部分得分之和

    def call(self, inputs):  # CRF本身不改变输出，它只是一个loss
        return inputs

    def loss(self, y_true, y_pred):  # 目标y_pred需要是one hot形式
        mask = 1-y_true[:, 1:, -1] if self.ignore_last_label else None
        y_true, y_pred = y_true[:, :,
                                :self.num_labels], y_pred[:, :, :self.num_labels]
        init_states = [y_pred[:, 0]]  # 初始状态
        log_norm, _, _ = tf.keras.backend.rnn(
            self.log_norm_step, y_pred[:, 1:], init_states, mask=mask)  # 计算Z向量（对数）
        log_norm = tf.reduce_logsumexp(log_norm, 1, keepdims=True)  # 计算Z（对数）
        path_score = self.path_score(y_pred, y_true)  # 计算分子（对数）
        return -(path_score - log_norm)  # 即log(分子/分母)

    def accuracy(self, y_true, y_pred):  # 训练过程中显示逐帧准确率的函数，排除了mask的影响
        mask = 1-y_true[:, :, -1] if self.ignore_last_label else None
        y_true, y_pred = y_true[:, :,
                                :self.num_labels], y_pred[:, :, :self.num_labels]
        isequal = tf.equal(tf.argmax(y_true, 2), tf.argmax(y_pred, 2))
        isequal = tf.cast(isequal, tf.float32)
        if mask == None:
            return tf.reduce_mean(isequal)
        else:
            return tf.reduce_sum(isequal*mask) / tf.reduce_sum(mask)

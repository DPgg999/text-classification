# coding: utf-8

import tensorflow as tf


class TextLstmAttentionConfig(object):
    """CNN配置参数"""

    embedding_dim = 64  # 词向量维度
    seq_length = 600  # 序列长度
    num_classes = 10  # 类别数
    num_filters = 256  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸
    vocab_size = 5000  # 词汇表达小

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard

    attention_size = hidden_dim # "注意力层，此参数和lstm隐藏层大小相同，双向则为其2倍"


class Text_Lstm_Attention_classfier(object):
    """文本分类，RNN模型"""
    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.rnn()

    def rnn(self):
        """rnn模型"""

        # 词向量映射
        with tf.device('/gpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)
            inputs=tf.unstack(embedding_inputs, axis=1)

        with tf.name_scope("rnn"):

            # lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim, forget_bias=1.0)  # 创建正向的cell
            # lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim, forget_bias=1.0)  # 创建反向的cell
            # 静态LSTM网络的构建
            lstm_cell = tf.contrib.rnn.LSTMCell(self.config.hidden_dim)
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=0.6)
            outputs, _ = tf.nn.static_rnn(lstm_cell, inputs, dtype=tf.float32)
        with tf.name_scope('attention'):
            # 定义attention layer
            attention_size = self.config.attention_size

            # outputs = tf.concat(_outputs, 2)
            # outputs = tf.transpose(outputs, [1, 0, 2])
            # last = outputs[-1]  # 取最后一个时序输出作为结果


            # attention_w = tf.Variable(tf.truncated_normal([2 * self.config.hidden_dim, attention_size], stddev=0.1),
            #                           name='attention_w')
            # attention_b = tf.Variable(tf.constant(0.1, shape=[attention_size]), name='attention_b')
            # u_list = []
            # # 以下部分为相似度的计算，对每个时间序列计算相似度
            # for t in range(self.config.seq_length):
            #     # print(tf.shape(outputs[t]))
            #     u_t = tf.tanh(tf.matmul(outputs[t], attention_w) + attention_b)
            #     u_list.append(u_t)
            # u_w = tf.Variable(tf.truncated_normal([attention_size, 1], stddev=0.1), name='attention_uw')
            # # 每个时间序列注意力权值的计算
            # attn_z = []
            # for t in range(self.config.seq_length):
            #     z_t = tf.matmul(u_list[t], u_w)
            #     attn_z.append(z_t)
            # # transform to batch_size * sequence_length
            # attn_zconcat = tf.concat(attn_z, axis=1)
            # alpha = tf.nn.softmax(attn_zconcat)
            # # 和注意力权值相乘得到注意力值
            # # transform to sequence_length * batch_size * 1 , same rank as outputs
            # alpha_trans = tf.reshape(tf.transpose(alpha, [1, 0]), [self.config.seq_length, -1, 1])
            # # 输出和权值相乘，使得重要的特征更加突出，不重要的减小其影响
            # final_output = tf.reduce_sum(outputs * alpha_trans, 0)

            # attention_w = tf.Variable(tf.truncated_normal([self.config.hidden_dim, attention_size], stddev=0.1),
            #                           name='attention_w')
            # attention_b = tf.Variable(tf.constant(0.1, shape=[attention_size]), name='attention_b')
            # u_list = []
            # # 以下部分为相似度的计算，对每个时间序列计算相似度
            # for t in range(self.config.seq_length):
            #     # print(tf.shape(outputs[t]))
            #     u_t = tf.tanh(tf.matmul(outputs[t], attention_w) + attention_b)
            #     u_list.append(u_t)
            # u_w = tf.Variable(tf.truncated_normal([attention_size, 1], stddev=0.1), name='attention_uw')
            # # 每个时间序列注意力权值的计算
            # attn_z = []
            # for t in range(self.config.seq_length):
            #     z_t = tf.matmul(u_list[t], u_w)
            #     attn_z.append(z_t)
            # # transform to batch_size * sequence_length
            # attn_zconcat = tf.concat(attn_z, axis=1)
            # alpha = tf.nn.softmax(attn_zconcat)
            # # 和注意力权值相乘得到注意力值
            # # transform to sequence_length * batch_size * 1 , same rank as outputs
            # alpha_trans = tf.reshape(tf.transpose(alpha, [1, 0]), [self.config.seq_length, -1, 1])
            # # 输出和权值相乘，使得重要的特征更加突出，不重要的减小其影响
            # final_output = tf.reduce_sum(outputs * alpha_trans, 0)

            attention_w = tf.Variable(tf.truncated_normal([self.config.hidden_dim, attention_size], stddev=0.1),
                                      name='attention_w')
            attention_b = tf.Variable(tf.constant(0.1, shape=[attention_size]), name='attention_b')
            u_list = []
            # 以下部分为相似度的计算，对每个时间序列计算相似度
            for t in range(self.config.hidden_dim):
                # print(tf.shape(outputs[t]))
                u_t = tf.tanh(tf.matmul(outputs[t], attention_w) + attention_b)
                u_list.append(u_t)
            u_w = tf.Variable(tf.truncated_normal([attention_size, 1], stddev=0.1), name='attention_uw')
            # 每个时间序列注意力权值的计算
            attn_z = []
            for t in range(self.config.seq_length):
                z_t = tf.matmul(u_list[t], u_w)
                attn_z.append(z_t)
            # transform to batch_size * sequence_length
            attn_zconcat = tf.concat(attn_z, axis=1)
            alpha = tf.nn.softmax(attn_zconcat)
            # 和注意力权值相乘得到注意力值
            # transform to sequence_length * batch_size * 1 , same rank as outputs
            alpha_trans = tf.reshape(tf.transpose(alpha, [1, 0]), [self.config.seq_length, -1, 1])
            # 输出和权值相乘，使得重要的特征更加突出，不重要的减小其影响
            final_output = tf.reduce_sum(outputs * alpha_trans, 0)

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            # fc = tf.layers.dense(last, self.config.hidden_dim, name='fc1')
            # fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            # fc = tf.nn.relu(fc)
            fc_w = tf.Variable(tf.truncated_normal([self.config.hidden_dim, self.config.num_classes], stddev=0.1), name='fc_w')
            fc_b = tf.Variable(tf.zeros([self.config.num_classes]), name='fc_b')

            # 分类器
            # self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            # self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别
            # 用于分类任务, outputs取最终一个时刻的输出
            pred = tf.matmul(final_output, fc_w) + fc_b
            self.logits = pred

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            # 用精度评估模型
            correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(self.input_y, 1))
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            # correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            # self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

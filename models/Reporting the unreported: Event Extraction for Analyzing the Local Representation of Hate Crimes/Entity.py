import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from preprocess import *


class Entity():
    def __init__(self, params, vocab, my_embeddings=None):
        self.params = params
        self.vocab = vocab
        for key in params:
            setattr(self, key, params[key])
        if self.pretrain:
            self.my_embeddings = my_embeddings

    def build_embedding(self):
        if self.pretrain:
            embedding_placeholder = tf.Variable(tf.constant(0.0, shape=[len(self.vocab),
                                                self.embedding_size]), trainable=False, name="W")
        else:
            embedding_placeholder = tf.get_variable("embedding",
                                                    initializer=tf.random_uniform(
                                                        [len(self.vocab), self.embedding_size], -1, 1),
                                                    dtype=tf.float32)
        return embedding_placeholder

    def build(self):
        # input data is in form of [batch_size, article_len, sentence_len]
        self.train_inputs = tf.placeholder(tf.int32, shape=[None, None, None], name="inputs")
        self.embedding_placeholder = self.build_embedding()
        self.indices = tf.placeholder(tf.int32, shape=[None, None, None])
        self.important_inputs = tf.gather_nd(self.train_inputs, self.indices)

        # length of each sentence in the whole batch
        self.sequence_length = tf.placeholder(tf.int64, [None, None])
        self.important_lengths = tf.gather_nd(self.sequence_length, self.indices)

        self.embed = tf.nn.embedding_lookup(self.embedding_placeholder, self.important_inputs)

        self.keep_ratio = tf.placeholder(tf.float32)

        # target labels corresponding to each article. Shape: [batch_size]
        self.target_group = tf.placeholder(tf.int64, [None])
        # the weight of each target label is 1 - (label frequency) / (all articles)
        self.target_weight = tf.placeholder(tf.float64, [None])

        # action labels corresponding to each article. Shape: [batch_size]
        self.hate_act = tf.placeholder(tf.int64, [None])
        # the weight of each action label is 1 - (label frequency) / (all articles)
        self.act_weight = tf.placeholder(tf.float64, [None])

        f_cell = tf.contrib.rnn.GRUCell(num_units=self.hidden_size)
        f_cell_drop = tf.contrib.rnn.DropoutWrapper(f_cell, input_keep_prob=self.keep_ratio)
        self.f_network = tf.contrib.rnn.MultiRNNCell([f_cell_drop] * self.num_layers)

        b_cell = tf.contrib.rnn.GRUCell(num_units=self.hidden_size, reuse=False )
        b_cell_drop = tf.contrib.rnn.DropoutWrapper(b_cell, input_keep_prob=self.keep_ratio)
        self.b_network = tf.contrib.rnn.MultiRNNCell([b_cell_drop] * self.num_layers)
        shape = tf.shape(self.embed)

        # the inputs are reshaped to [all sentences, sentence_len] to be passed to LSTM
        embed = tf.reshape(self.embed, [shape[0] * shape[1], shape[2], self.embedding_size])
        self.important_lengths = tf.reshape(self.important_lengths, [tf.shape(embed)[0]])

        # Bi-directional LSTM to capture the sentence representation
        bi_outputs, bi_states = tf.nn.bidirectional_dynamic_rnn(self.f_network, self.b_network, embed,
                                                                dtype=tf.float32,
                                                                sequence_length=self.important_lengths)
        fw_states, bw_states = bi_states
        state = tf.concat([fw_states, bw_states], 2)

        # vectors are reshaped to form the articles
        state = tf.reshape(state, [shape[0], shape[1], 2 * self.hidden_size])
        state = tf.nn.dropout(state, keep_prob=self.keep_ratio)

        fc_target = fully_connected(state, 9)
        fc_act = fully_connected(state, 6)

        self.high_target = tf.reduce_max(fc_target, axis=[1])
        self.high_act = tf.reduce_max(fc_act, axis=[1])

        t_weight = tf.gather(self.target_weight, self.target_group)
        a_weight = tf.gather(self.act_weight, self.hate_act)

        # weighted losses are calculated
        self.target_xentropy = tf.losses.sparse_softmax_cross_entropy(labels=self.target_group,
                                                                      logits=self.high_target,
                                                                      weights=t_weight)
        self.act_xentropy = tf.losses.sparse_softmax_cross_entropy(labels=self.hate_act,
                                                                   logits=self.high_act,
                                                                   weights=a_weight)

        self.loss = tf.add(self.target_xentropy, self.act_xentropy)
        self.predicted_target = tf.argmax(self.high_target, 1)
        self.predicted_act = tf.argmax(self.high_act, 1)

        self.accuracy_target = tf.reduce_mean(
              tf.cast(tf.equal(self.predicted_target, self.target_group), tf.float32))
        self.accuracy_act = tf.reduce_mean(
            tf.cast(tf.equal(self.predicted_act, self.hate_act), tf.float32))

        self.accuracy = (self.accuracy_target + self.accuracy_act) / 2
        self.training_op = tf.train.AdamOptimizer(learning_rate=self.entity_learning_rate).minimize(self.loss)

    def get_feed_dict(self, batch, weights, train=True):
        target_weight, act_weight = weights
        indices = np.array([[[idx, b["best_sent"][0]], [idx, b["best_sent"][1]]]
                            for idx, b in enumerate(batch)])

        feed_dict = {self.train_inputs: np.array([b["article"] for b in batch]),
                 self.sequence_length: np.array([b["lengths"] for b in batch]),
                 self.keep_ratio: self.entity_keep_ratio if train else 1,
                 self.target_weight: target_weight,
                 self.act_weight: act_weight,
                 self.indices: indices
                 }
        if train:
            feed_dict[self.target_group] = np.array([b["target_label"] for b in batch])
            feed_dict[self.hate_act] = np.array([b["action_label"] for b in batch])

        if self.pretrain:
            feed_dict[self.embedding_placeholder] = self.my_embeddings
        return feed_dict

    def predict(self, unlabeled_batches, weights):
        target_pred = list()
        action_pred = list()
        saver = tf.train.Saver()
        with tf.Session() as self.sess:
            saver.restore(self.sess, "model/Entity/entity_model_2.ckpt")
            for i in range(len(unlabeled_batches) // 5000 + 1):
                print("Gathering labels for 500 datapoints, batch #", i)
                sub = unlabeled_batches[i * 5000: min((i + 1) * 5000, len(unlabeled_batches))]
                batches = BatchIt(sub, self.batch_size, self.vocab, True)
                for batch in batches:
                    feed_dict = self.get_feed_dict(batch, weights, False)
                    target_, act_ = self.sess.run([self.predicted_target, self.predicted_act], feed_dict=feed_dict)
                    target_pred.extend(list(target_))
                    action_pred.extend(list(act_))
        return target_pred, action_pred


    def run_model(self, batches, dev_batches, test_batches, weights):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as self.sess:
            self.sess.run(init)
            epoch = 1
            while True:
                ## Train
                epoch_loss = float(0)
                train_accuracy = 0
                for batch in batches:
                    feed_dict = self.get_feed_dict(batch, weights)
                    loss_val, _= self.sess.run([self.loss, self.training_op], feed_dict= feed_dict)
                    train_accuracy += self.accuracy.eval(feed_dict=feed_dict)
                    epoch_loss += loss_val
                ## Dev
                dev_accuracy = 0
                for batch in dev_batches:
                    feed_dict = self.get_feed_dict(batch, weights)
                    dev_accuracy += self.accuracy.eval(feed_dict=feed_dict)
                print(epoch, "Train accuracy:", train_accuracy / len(batches),
                      "Loss: ", epoch_loss / float(len(batches)),
                      "Dev accuracy: ", dev_accuracy / len(dev_batches))
                if epoch == self.epochs:
                    save_path = saver.save(self.sess, "model/Entity/entity_model_2.ckpt")
                    break
                epoch += 1
            ## Test
            t_pred, a_pred, t_true, a_true = list(), list(), list(), list()
            for batch in test_batches:
                feed_dict = self.get_feed_dict(batch, weights)
                dev_accuracy += self.accuracy.eval(feed_dict=feed_dict)
                try:
                    target_, act_ = self.sess.run([self.predicted_target, self.predicted_act], feed_dict=feed_dict)
                    t_pred.extend(list(target_))
                    a_pred.extend(list(act_))
                    t_true.extend([b["target_label"] for b in batch])
                    a_true.extend([b["action_label"] for b in batch])
                except Exception:
                    print()
            print("Target F1 score: ", f1_score(t_true, t_pred, average="weighted"),
                  "Target Precision: ", precision_score(t_true, t_pred, average="weighted"),
                  "Target Recall:", recall_score(t_true, t_pred, average="weighted"), "\n",
                  "Act F1 score: ", f1_score(a_true, a_pred, average="weighted"),
                  "Act Precision: ", precision_score(a_true, a_pred, average="weighted"),
                  "Act Recall:", recall_score(a_true, a_pred, average="weighted")
                  )
        return

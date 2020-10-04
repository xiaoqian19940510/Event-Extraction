import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from sklearn.metrics import f1_score, precision_score, recall_score
from preprocess import *

class Hi_Attn():
    def __init__(self, params, vocabs, my_embeddings=None):
        self.params = params
        self.vocabs = vocabs
        for key in params:
            setattr(self, key, params[key])
        if self.pretrain:
            self.my_embeddings = my_embeddings

    def build_embedding(self):
        if self.pretrain:
            embedding_placeholder = tf.placeholder(tf.float32, [len(self.vocabs), self.embedding_size])
        else:
            embedding_placeholder = tf.get_variable("embedding", initializer=tf.random_uniform(
                [len(self.vocabs), self.embedding_size], -1, 1), dtype=tf.float32)
        return embedding_placeholder


    def build(self):
        tf.reset_default_graph()
        # length of each sentence in the whole batch
        self.sequence_length = tf.placeholder(tf.int64, [None])
        self.article_lens = tf.placeholder(tf.int64, [None])

        # input data is in form of [batch_size, article_len, sentence_len]
        self.train_inputs = tf.placeholder(tf.int32, shape=[None, None, None])
        self.embedding_placeholder = self.build_embedding()
        self.embed = tf.nn.embedding_lookup(self.embedding_placeholder, self.train_inputs)
        self.output = tf.placeholder(tf.int64, [None])
        self.keep_prob = tf.placeholder(tf.float32)

        shape = tf.shape(self.embed)
        embed = tf.reshape(self.embed, [shape[0] * shape[1], shape[2], self.embedding_size])
        self.sequence_length = tf.reshape(self.sequence_length, [tf.shape(embed)[0]])

        f_cell = tf.contrib.rnn.GRUCell(num_units=self.hidden_size)
        #f_cell_drop = tf.contrib.rnn.DropoutWrapper(f_cell, input_keep_prob=self.keep_prob)
        self.f_network = tf.contrib.rnn.MultiRNNCell([f_cell] * self.num_layers)

        b_cell = tf.contrib.rnn.GRUCell(num_units=self.hidden_size, reuse=False)
        #b_cell_drop = tf.contrib.rnn.DropoutWrapper(b_cell, input_keep_prob=self.keep_prob)
        self.b_network = tf.contrib.rnn.MultiRNNCell([b_cell] * self.num_layers)

        ############ WORD LEVEL ATTENTION ################
        # the inputs are reshaped to [all sentences, sentence_len] to be passed to LSTM
        bi_outputs, bi_states = tf.nn.bidirectional_dynamic_rnn(self.f_network, self.b_network,
                                                                embed,dtype=tf.float32,
                                                                sequence_length=self.sequence_length,
                                                                scope="word")
        fw_outputs, bw_outputs = bi_outputs
        state = tf.concat([fw_outputs, bw_outputs], 2)
        state = tf.reshape(state, [shape[0], shape[1], shape[2], 2 * self.hidden_size])

        self.attn = tf.tanh(fully_connected(state, self.attention_size))
        self.alphas = tf.nn.softmax(tf.layers.dense(self.attn, self.attention_size, use_bias=False))
        word_attn = tf.reduce_sum(state * self.alphas, 2)
        drop = tf.nn.dropout(word_attn, self.keep_prob)
        #[Batch, num_sentences]
        ############################################################

        ############ SENTENCE LEVEL ATTENTION ################
        sent_cell = tf.contrib.rnn.GRUCell(num_units=self.hidden_size, reuse=False)
        #sent_cell_drop = tf.contrib.rnn.DropoutWrapper(sent_cell, input_keep_prob=self.keep_prob)
        self.sent_network = tf.contrib.rnn.MultiRNNCell([sent_cell] * self.num_layers)

        word_attn = tf.reshape(drop, [shape[0], shape[1], 2 * self.hidden_size])
        sent_bi_outputs, sent_bi_states = tf.nn.bidirectional_dynamic_rnn(self.sent_network, self.sent_network,
                                                                word_attn, dtype=tf.float32,
                                                                sequence_length=self.article_lens,
                                                                scope="sentence")
        sent_fw_outputs, sent_bw_outputs = sent_bi_outputs
        sent_state = tf.concat([sent_fw_outputs, sent_bw_outputs], 2)
        sent_state = tf.reshape(sent_state, [shape[0], shape[1], 2 * self.hidden_size])

        sent_attn = tf.tanh(fully_connected(sent_state, self.attention_size))
        self.sent_alphas = tf.nn.softmax(tf.layers.dense(sent_attn, self.attention_size, use_bias=False))
        self.final_attn = tf.reduce_sum(sent_state * self.sent_alphas, 1)

        self.logits = fully_connected(self.final_attn, 2)
        logits = tf.reshape(self.logits, [shape[0], 2])

        self.xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.output,
                                                                       logits=logits)
        self.loss = tf.reduce_mean(self.xentropy)
        self.predicted_label = tf.argmax(self.logits, 1)

        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(self.predicted_label, self.output), tf.float32))
        self.training_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


    def predict(self, unlabeled_batches):
        hate_pred = list()
        indices_pred = list()
        saver = tf.train.Saver()
        with tf.Session() as self.sess:
            saver.restore(self.sess,  "model/Hi-Attn/attn_model.ckpt")
            for i in range(len(unlabeled_batches) // 5000 + 1):
                print("Gathering labels for 5000 datapoints, batch #", i)
                sub = unlabeled_batches[i * 5000: min((i + 1) * 5000, len(unlabeled_batches))]
                batches = BatchIt(sub, self.batch_size, self.vocabs, True)
                for batch in batches:
                    feed_dict = {self.train_inputs: np.array([b[0] for b in batch]),
                                 self.sequence_length: np.array([l for b in batch for l in b[1]]),
                                 self.keep_prob: 1,
                                 self.article_lens: np.array([b[2] for b in batch])
                                 }
                    if self.pretrain:
                        feed_dict[self.embedding_placeholder] = self.my_embeddings
                    hate = self.sess.run(self.predicted_label,
                                                  feed_dict=feed_dict)
                    hate_pred.extend(list(hate))
        return hate_pred, []

    def get_feed_dict(self, batch, train=True):
        feed_dict = {self.train_inputs: np.array([b[0] for b in batch]),
                     self.sequence_length: np.array([l for b in batch for l in b[1]]),
                     self.keep_prob: self.keep_ratio if train else 1,
                     self.output: np.array([b[2] for b in batch]),
                     self.article_lens: np.array([b[5] for b in batch])
                     }
        if self.pretrain:
            feed_dict[self.embedding_placeholder] = self.my_embeddings
        return feed_dict

    def run_model(self, batches, dev_batches, test_batches):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as self.sess:
            # init.run()
            self.sess.run(init)
            epoch = 1
            while True:
                ## Train
                epoch_loss = float(0)
                epoch += 1
                train_accuracy = 0
                for batch in batches:
                    feed_dict = self.get_feed_dict(batch)
                    loss_val, _, log = self.sess.run([self.loss, self.training_op, self.logits],
                                                     feed_dict=feed_dict)
                    train_accuracy += self.accuracy.eval(feed_dict=feed_dict)
                    epoch_loss += loss_val
                ## Dev
                test_accuracy = 0
                hate_pred, hate_true = list(), list()
                for batch in dev_batches:
                    feed_dict = self.get_feed_dict(batch, False)
                    test_accuracy += self.accuracy.eval(feed_dict=feed_dict)
                    try:
                        hate = self.predicted_label.eval(feed_dict=feed_dict)
                        hate_pred.extend(list(hate))
                        hate_true.extend([b[2] for b in batch])
                    except Exception:
                        print()
                print(sum(hate_pred))
                print(epoch, "Train accuracy:", train_accuracy / len(batches),
                      "Loss:", epoch_loss / float(len(batches)),
                      "Test accuracy:", test_accuracy / len(dev_batches),
                      "Hate F1:", f1_score(hate_true, hate_pred, average="binary"),
                      "Precision", precision_score(hate_true, hate_pred),
                      "Recall", recall_score(hate_true, hate_pred))
                if epoch == self.epochs:
                    save_path = saver.save(self.sess, "model/Hi-Attn/attn_model.ckpt")
                    break

            test_accuracy = 0
            for batch in test_batches:
                feed_dict = self.get_feed_dict(batch, False)
                test_accuracy += self.accuracy.eval(feed_dict=feed_dict)
                try:
                    hate = self.predicted_label.eval(feed_dict=feed_dict)
                    hate_pred.extend(list(hate))
                    hate_true.extend([b[2] for b in batch])
                except Exception:
                    print()
            print("Test report",
                  "Test accuracy:", test_accuracy / len(test_batches),
                  "Hate F1:", f1_score(hate_true, hate_pred, average="binary"),
                  "Precision", precision_score(hate_true, hate_pred),
                  "Recall", recall_score(hate_true, hate_pred))
        return [], [], []

# =========================================================================================
# Implementation of "Show, Attend and Tell: Neural Caption Generator With Visual Attention".
# There are some notations.
# N is batch size.
# L is spacial size of feature vector (196).
# D is dimension of image feature vector (512).
# T is the number of time step which is equal to caption's length-1 (16).
# V is vocabulary size (about 10000).
# M is dimension of word vector which is embedding size (default is 512).
# H is dimension of hidden state (default is 1024).
# =========================================================================================

from __future__ import division

import tensorflow as tf
import numpy as np

class CaptionGenerator(object):
    def __init__(self, word_to_idx, dim_feature=[196, 512], dim_embed=512, dim_hidden=1024, n_time_step=16,
                  prev2out=True, ctx2out=True, alpha_c=0.0, selector=True, dropout=True):
        """
        Args:
            word_to_idx: word-to-index mapping dictionary.
            dim_feature: (optional) Dimension of vggnet19 conv5_3 feature vectors.
            dim_embed: (optional) Dimension of word embedding.
            dim_hidden: (optional) Dimension of all hidden state.
            n_time_step: (optional) Time step size of LSTM.
            prev2out: (optional) previously generated word to hidden state. (see Eq (7) for explanation)
            ctx2out: (optional) context to hidden state (see Eq (7) for explanation)
            alpha_c: (optional) Doubly stochastic regularization coefficient. (see Section (4.2.1) for explanation)
            selector: (optional) gating scalar for context vector. (see Section (4.2.1) for explanation)
            dropout: (optional) If true then dropout layer is added.
        """

        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
        self.prev2out = prev2out
        self.ctx2out = ctx2out
        self.alpha_c = alpha_c
        self.selector = selector
        self.dropout = dropout
        self.V = len(word_to_idx)
        self.L = dim_feature[0]
        self.D = dim_feature[1]
        self.M = dim_embed
        self.H = dim_hidden
        self.T = n_time_step
        self._start = word_to_idx['<START>']
        self._null = word_to_idx['<NULL>']

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        # Place holder for features and captions
        self.features = tf.placeholder(tf.float32, [None, self.L, self.D])
        self.captions = tf.placeholder(tf.int32, [None, self.T + 1])
        self.generated_caption = tf.placeholder(tf.int32, [None, self.T])
        self.given_num = tf.placeholder(tf.int32, shape=())

    def _get_initial_lstm(self, features):
        with tf.variable_scope('initial_lstm'):
            features_mean = tf.reduce_mean(features, 1)

            w_h = tf.get_variable('w_h', [self.D, self.H], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.H], initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

            w_c = tf.get_variable('w_c', [self.D, self.H], initializer=self.weight_initializer)
            b_c = tf.get_variable('b_c', [self.H], initializer=self.const_initializer)
            c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
            return c, h

    def _word_embedding(self, inputs, reuse=False):
        with tf.variable_scope('word_embedding', reuse=reuse):
            w = tf.get_variable('w', [self.V, self.M], initializer=self.emb_initializer)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
            return x

    def _project_features(self, features):
        with tf.variable_scope('project_features'):
            w = tf.get_variable('w', [self.D, self.D], initializer=self.weight_initializer)
            features_flat = tf.reshape(features, [-1, self.D])
            features_proj = tf.matmul(features_flat, w)
            features_proj = tf.reshape(features_proj, [-1, self.L, self.D])
            return features_proj

    def _attention_layer(self, features, features_proj, h, reuse=False):
        with tf.variable_scope('attention_layer', reuse=reuse):
            w = tf.get_variable('w', [self.H, self.D], initializer=self.weight_initializer)
            b = tf.get_variable('b', [self.D], initializer=self.const_initializer)
            w_att = tf.get_variable('w_att', [self.D, 1], initializer=self.weight_initializer)

            h_att = tf.nn.relu(features_proj + tf.expand_dims(tf.matmul(h, w), 1) + b)    # (N, L, D)
            out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att), [-1, self.L])   # (N, L)
            alpha = tf.nn.softmax(out_att)
            context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')   #(N, D)
            return context, alpha

    def _selector(self, context, h, reuse=False):
        with tf.variable_scope('selector', reuse=reuse):
            w = tf.get_variable('w', [self.H, 1], initializer=self.weight_initializer)
            b = tf.get_variable('b', [1], initializer=self.const_initializer)
            beta = tf.nn.sigmoid(tf.matmul(h, w) + b, 'beta')    # (N, 1)
            context = tf.multiply(beta, context, name='selected_context')
            return context, beta

    def _decode_lstm(self, x, h, context, dropout=False, reuse=False):
        with tf.variable_scope('logits', reuse=reuse):
            w_h = tf.get_variable('w_h', [self.H, self.M], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.M], initializer=self.const_initializer)
            w_out = tf.get_variable('w_out', [self.M, self.V], initializer=self.weight_initializer)
            b_out = tf.get_variable('b_out', [self.V], initializer=self.const_initializer)

            if dropout:
                h = tf.nn.dropout(h, 0.5)
            h_logits = tf.matmul(h, w_h) + b_h

            if self.ctx2out:
                w_ctx2out = tf.get_variable('w_ctx2out', [self.D, self.M], initializer=self.weight_initializer)
                h_logits += tf.matmul(context, w_ctx2out)

            if self.prev2out:
                h_logits += x
            h_logits = tf.nn.tanh(h_logits)

            if dropout:
                h_logits = tf.nn.dropout(h_logits, 0.5)
            out_logits = tf.matmul(h_logits, w_out) + b_out
            return out_logits

    def _batch_norm(self, x, mode='train', name=None):
        return tf.contrib.layers.batch_norm(inputs=x,
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=(mode=='train'),
                                            updates_collections=None,
                                            scope=(name+'batch_norm'))

    def build_model(self):
        features = self.features
        captions = self.captions
        batch_size = tf.shape(features)[0]

        captions_in = captions[:, :self.T]
        captions_out = captions[:, 1:]
        mask = tf.to_float(tf.not_equal(captions_out, self._null))

        self.rewards = tf.placeholder(tf.float32, shape=[None, self.T]) # get from rollout policy and discriminator

        # batch normalize feature vectors
        features = self._batch_norm(features, mode='train', name='conv_features')

        c, h = self._get_initial_lstm(features=features)
        x = self._word_embedding(inputs=captions_in)
        features_proj = self._project_features(features=features)

        loss = 0.0
        alpha_list = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

        loss_list = []


        for t in range(self.T):
            context, alpha = self._attention_layer(features, features_proj, h, reuse=(t!=0))
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, reuse=(t!=0))

            with tf.variable_scope('lstm', reuse=(t!=0)):
                _, (c, h) = lstm_cell(inputs=tf.concat([x[:,t,:], context], 1), state=[c, h])

            logits = self._decode_lstm(x[:,t,:], h, context, dropout=self.dropout, reuse=(t!=0))
            losses = (tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=captions_out[:, t]) * mask[:, t])
            loss_list.append(tf.reshape(losses, [-1]))

        loss_list = tf.transpose(tf.stack(loss_list), (1, 0))

        loss = tf.reduce_sum(loss_list)
        g_loss = tf.reduce_sum(tf.reshape(loss_list, [-1]) * tf.reshape(self.rewards, [-1]))

        if self.alpha_c > 0:
            alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))     # (N, T, L)
            alphas_all = tf.reduce_sum(alphas, 1)      # (N, L)
            alpha_reg = self.alpha_c * tf.reduce_sum((16./196 - alphas_all) ** 2)
            loss += alpha_reg
            g_loss += alpha_reg

        return loss / tf.to_float(batch_size), g_loss / tf.to_float(batch_size)

    def build_sampler(self, max_len=20):
        features = self.features

        # batch normalize feature vectors
        features = self._batch_norm(features, mode='test', name='conv_features')

        c, h = self._get_initial_lstm(features=features)
        features_proj = self._project_features(features=features)

        sampled_word_list = []
        alpha_list = []
        beta_list = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

        for t in range(max_len):
            if t == 0:
                x = self._word_embedding(inputs=tf.fill([tf.shape(features)[0]], self._start))
            else:
                x = self._word_embedding(inputs=sampled_word, reuse=True)

            context, alpha = self._attention_layer(features, features_proj, h, reuse=(t!=0))
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, reuse=(t!=0))
                beta_list.append(beta)

            with tf.variable_scope('lstm', reuse=(t!=0)):
                _, (c, h) = lstm_cell(inputs=tf.concat([x, context], 1), state=[c, h])

            logits = self._decode_lstm(x, h, context, reuse=(t!=0))
            sampled_word = tf.argmax(logits, 1)
            sampled_word_list.append(sampled_word)

        alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))     # (N, T, L)
        betas = tf.transpose(tf.squeeze(beta_list), (1, 0))    # (N, T)
        sampled_captions = tf.transpose(tf.stack(sampled_word_list), (1, 0))     # (N, max_len)
        return alphas, betas, sampled_captions


    def build_rollout(self, max_len=20):
        features = self.features
        generated_caption = self.generated_caption
        # batch normalize feature vectors
        features = self._batch_norm(features, mode='test', name='conv_features')
        given_num = self.given_num

        c, h = self._get_initial_lstm(features=features)
        features_proj = self._project_features(features=features)

        sampled_word_list = []

        generated_caption = tf.cast(tf.transpose(generated_caption, (1, 0)), dtype=tf.int64)

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

        def recurrence_before(t, c, h, sampled_word, given_num, sampled_word_list):
            sampled_word_list = sampled_word_list.write(t - 1, tf.gather(generated_caption, t - 1))
            x = self._word_embedding(inputs=tf.gather(generated_caption, t - 1), reuse=True)
            context, alpha = self._attention_layer(features, features_proj, h, reuse=True)
            if self.selector:
                context, beta = self._selector(context, h, reuse=True)
            with tf.variable_scope('lstm', reuse=True):
                _, (c, h) = lstm_cell(inputs=tf.concat([x, context], 1), state=[c, h])

            logits = self._decode_lstm(x, h, context, reuse=True)
            sampled_word = tf.argmax(logits, 1)
            return t + 1, c, h, sampled_word, given_num, sampled_word_list

        def recurrence_after(t, c, h, sampled_word, given_num, sampled_word_list):
            x = self._word_embedding(inputs=sampled_word, reuse=True)
            context, alpha = self._attention_layer(features, features_proj, h, reuse=True)
            if self.selector:
                context, beta = self._selector(context, h, reuse=True)
            with tf.variable_scope('lstm', reuse=True):
                _, (c, h) = lstm_cell(inputs=tf.concat([x, context], 1), state=[c, h])

            logits = self._decode_lstm(x, h, context, reuse=True)
            sampled_word = tf.argmax(logits, 1)
            sampled_word_list = sampled_word_list.write(t, sampled_word)
            return t + 1, c, h, sampled_word, given_num, sampled_word_list

        x = self._word_embedding(inputs=tf.fill([tf.shape(features)[0]], self._start))
        context, alpha = self._attention_layer(features, features_proj, h, reuse=False)

        if self.selector:
            context, beta = self._selector(context, h, reuse=False)

        with tf.variable_scope('lstm', reuse=False):
            _, (c, h) = lstm_cell(inputs=tf.concat([x, context], 1), state=[c, h])

        logits = self._decode_lstm(x, h, context, reuse=False)
        sampled_word = tf.argmax(logits, 1)

        sampled_word_list = tf.TensorArray(dtype=tf.int64, size=max_len,
                                     dynamic_size=False, infer_shape=True)

        t, c, h, sampled_word, given_num, sampled_word_list = tf.while_loop(
            cond=lambda t, _1, _2, _3, given_num, _4 : t <= given_num,
            body=recurrence_before,
            loop_vars=(tf.constant(1, dtype=tf.int32),
                        c, h, sampled_word, given_num, sampled_word_list)
        )

        sampled_word_list = sampled_word_list.write(t - 1, sampled_word)

        _, _, _, _, _, sampled_word_list = tf.while_loop(
            cond=lambda t, _1, _2, _3, given_num, _4 : t < max_len,
            body=recurrence_after,
            loop_vars=(t, c, h, sampled_word, given_num, sampled_word_list)
        )

        self.rolled_out_caption = tf.transpose(sampled_word_list.stack(), (1, 0))     # (N, max_len)
        return self.rolled_out_caption

    def fix_samples(self, samples):
        for generated_caption in samples:
            for i in xrange(len(generated_caption) - 1, -1, -1):
                if generated_caption[i] != 2:
                    if i < 15:
                        generated_caption[i + 1] = 2
                    break
                else:
                    generated_caption[i] = 0

    def get_rewards(self, sess, num_rollout, features, sampled_caption, discriminator, max_length=20):
        rewards = []
        for i in range(num_rollout):
            for given_num in range(1, max_length):
                feed = {self.features: features,
                self.generated_caption: sampled_caption,
                self.given_num: given_num}
                samples = sess.run(self.rolled_out_caption, feed)
                self.fix_samples(samples)
                feed = {discriminator.input_x: samples, discriminator.dropout_keep_prob: 1.0}
                ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
                ypred = np.array([item[1] for item in ypred_for_auc])
                if i == 0:
                    rewards.append(ypred)
                else:
                    rewards[given_num - 1] += ypred

            # the last token reward
            feed = {discriminator.input_x: sampled_caption, discriminator.dropout_keep_prob: 1.0}
            ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
            ypred = np.array([item[1] for item in ypred_for_auc])
            if i == 0:
                rewards.append(ypred)
            else:
                rewards[max_length - 1] += ypred

        rewards = np.transpose(np.array(rewards)) / (1.0 * num_rollout)  # batch_size x seq_length
        return rewards

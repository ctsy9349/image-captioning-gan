import tensorflow as tf
import numpy as np

class Discriminator(object):
	def __init__(self, word_to_idx, dim_feature=[196, 512], dim_embed=512, dim_hidden=1024, n_time_step=16,
	prev2out=True, ctx2out=True, alpha_c=0.0, selector=True, dropout=True, learning_rate=0.1):
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
		self.learning_rate = learning_rate
		self._start = word_to_idx['<START>']
		self._null = word_to_idx['<NULL>']

		self.weight_initializer = tf.contrib.layers.xavier_initializer()
		self.const_initializer = tf.constant_initializer(0.0)
		self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

		# Place holder for features and captions
		self.features = tf.placeholder(tf.float32, [None, self.L, self.D])
		self.captions = tf.placeholder(tf.float32, [None, self.T + 1, 1])
		self.target = tf.placeholder(tf.float32, [None, 1])
		self.loss = None

	def build_model(self):
		if self.loss is not None:
			return self.loss

		features = self.features
		captions = self.captions
		target = self.target
		with tf.variable_scope('d_lstm', reuse=tf.AUTO_REUSE):
			features_flat = tf.reshape(features, [-1, 196 * 512])
			features_dense = tf.layers.dense(inputs=features_flat, units=self.H, activation=tf.nn.relu, kernel_initializer=self.weight_initializer)

			cell = tf.nn.rnn_cell.LSTMCell(self.H,state_is_tuple=True)
			val, state = tf.nn.dynamic_rnn(cell, captions, dtype=tf.float32)

			val = tf.transpose(val, [1, 0, 2])
			last = tf.gather(val, int(val.get_shape()[0]) - 1)
			dot_prod = tf.reduce_sum( tf.multiply( last, features_dense ), 1, keep_dims=True )
			W = tf.Variable(tf.truncated_normal([1, 1]))
			b = tf.Variable(tf.constant(0.1, shape=[1]))
			prediction = tf.matmul(dot_prod, W) + b
			pred_sigmoid = tf.sigmoid(prediction)   # for prediction
			x_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=target)
			loss = tf.reduce_mean(x_entropy)
		self.pred_sigmoid = pred_sigmoid
		self.loss = loss
		self.train_step = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
		self.dot_prod = dot_prod
		self.last = last
		self.features_dense = features_dense
		self.features_flat = features_flat
		return loss

	def train(self, sess, e, i, image_features, image_captions, y):
		features = self.features
		captions = self.captions
		image_captions = np.reshape(image_captions, (image_captions.shape[0], image_captions.shape[1], 1))
		target = self.target
		loss = self.loss
		train_step = self.train_step
		fd_train = {features: image_features, captions: image_captions, target: y}
		print "Dotprod:", (self.dot_prod.eval(fd_train)[:10, :])
		print "LSTM:",(self.last.eval(fd_train)[:, :10])
		print "CNN:", (self.features_dense.eval(fd_train)[:, :10])
		print "Feat:", (self.features_flat.eval(fd_train)[:, :10])
		train_step.run(fd_train)
		loss_step = loss.eval(fd_train)
		print('Epoch %6d, Step %6d: Loss = %8.3f' % (e, i, loss_step))
		return loss_step

	def save(self):
		pass

	def get_rewards(self, image_features, image_captions):
		features = self.features
		captions = self.captions
		target = self.target
		loss = self.loss
		pred_sigmoid = self.pred_sigmoid
		image_captions = np.reshape(image_captions, (image_captions.shape[0], image_captions.shape[1], 1))
		print 'Making discriminator predictions ...'
		fd_train = {features: image_features, captions: image_captions}
		print (self.dot_prod.eval(fd_train))
		print (self.last.eval(fd_train)[:, :20])
		print (self.features_dense.eval(fd_train)[:, :20])
		print (self.features_flat.eval(fd_train)[:, :20])
		pred = pred_sigmoid.eval(fd_train)
		return pred

	def get_grads(self, rewards):
		pass

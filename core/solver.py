import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
import time, sys
import os
import cPickle as pickle
from scipy import ndimage
from utils import *
from bleu import evaluate
from core.data_provider import DDataProvider

class CaptioningSolver(object):
	def __init__(self, model, discriminator, data, val_data, **kwargs):
		"""
		Required Arguments:
		- model: Show Attend and Tell caption generating model
		- data: Training data; dictionary with the following keys:
		- features: Feature vectors of shape (82783, 196, 512)
		- file_names: Image file names of shape (82783, )
		- captions: Captions of shape (400000, 17)
		- image_idxs: Indices for mapping caption to image of shape (400000, )
		- word_to_idx: Mapping dictionary from word to index
		- val_data: validation data; for print out BLEU scores for each epoch.
		Optional Arguments:
		- n_epochs: The number of epochs to run for training.
		- batch_size: Mini batch size.
		- update_rule: A string giving the name of an update rule
		- learning_rate: Learning rate; default value is 0.01.
		- print_every: Integer; training losses will be printed every print_every iterations.
		- save_every: Integer; model variables will be saved every save_every epoch.
		- pretrained_model: String; pretrained model path
		- model_path: String; model path for saving
		- test_model: String; model path for test
		"""

		self.model = model
		self.discriminator = discriminator
		self.data = data
		self.val_data = val_data
		self.n_epochs = kwargs.pop('n_epochs', 10)
		self.batch_size = kwargs.pop('batch_size', 100)
		self.update_rule = kwargs.pop('update_rule', 'adam')
		self.learning_rate = kwargs.pop('learning_rate', 0.01)
		self.print_bleu = kwargs.pop('print_bleu', False)
		self.print_every = kwargs.pop('print_every', 100)
		self.save_every = kwargs.pop('save_every', 1)
		self.log_path = kwargs.pop('log_path', './log/')
		self.model_path = kwargs.pop('model_path', './model/')
		self.pretrained_model = kwargs.pop('pretrained_model', None)
		self.test_model = kwargs.pop('test_model', './model/lstm/model-1')
		self.train_new = kwargs.pop('train_new', './model/lstm/model-20')
		self.gpu_list = kwargs.pop('gpu_list', None)
		self.dis_dropout_keep_prob = kwargs.pop('gpu_list', 0.75)
		self.num_rollout = kwargs.pop('num_rollout', 20)

		# set an optimizer by update rule
		if self.update_rule == 'adam':
			self.optimizer = tf.train.AdamOptimizer
		elif self.update_rule == 'momentum':
			self.optimizer = tf.train.MomentumOptimizer
		elif self.update_rule == 'rmsprop':
			self.optimizer = tf.train.RMSPropOptimizer

		if not os.path.exists(self.model_path):
			os.makedirs(self.model_path)
		if not os.path.exists(self.log_path):
			os.makedirs(self.log_path)

	def fix_wrong_captions(self, wrong_capt_batch, all_captions, rand_id, rand_id_c):
		n_examples = all_captions.shape[0]
		counter = 0
		for i, j in zip(rand_id, rand_id_c):
			if i == j:
				ind = j + 1
				wrong_capt_batch[counter] = all_captions[ind]
				counter += 1

	def test_discrim(self):
		n_examples = self.data['captions'].shape[0]
		n_iters_per_epoch = int(np.ceil(float(n_examples)/self.batch_size))
		features = self.data['features']
		captions = self.data['captions']
		image_idxs = self.data['image_idxs']
		val_features = self.val_data['features']
		n_iters_val = int(np.ceil(float(val_features.shape[0])/self.batch_size))

		print "Data size: %d" %n_examples
		print "Batch size: %d" %self.batch_size
		print "Iterations per epoch: %d" %n_iters_per_epoch

		config = tf.ConfigProto(allow_soft_placement = True)
		config.gpu_options.allow_growth = True
		if self.gpu_list is not None:
			config.gpu_options.visible_device_list = self.gpu_list

		"""
		Testing Discriminator
		"""
		print "\n\nTesting Discriminator ...\n"
		prev_loss = -1
		curr_loss = 0

		# build a graph to sample captions
		alphas, betas, sampled_captions = self.model.build_sampler(max_len=16)    # (N, max_len, L), (N, max_len)

		with tf.Session(config=config) as sess:
			tf.initialize_all_variables().run()

			# Different Loading Paths
			saver = tf.train.Saver()#var_list = non_d_vars)
			saver.restore(sess, self.test_model)

			# Removing <START> token from original captions
			original_captions = captions[:, 1:]

			for e in range(self.n_epochs):
				# Getting New Training Data
				# print "\n\nEpoch:", e
				generated_captions = self.get_generated_captions(sess, alphas, betas, sampled_captions, features, n_iters_per_epoch)
				data_provider = DDataProvider(original_captions, generated_captions)
				all_captions, all_labels = data_provider.get_data()
				for i in range(n_iters_per_epoch):
					captions_batch = all_captions[i*self.batch_size:(i+1)*self.batch_size]
					labels_batch = all_labels[i*self.batch_size:(i+1)*self.batch_size]
					feed_dict_discrim = {
						self.discriminator.input_x: captions_batch,
						self.discriminator.dropout_keep_prob: 1.0
					}
					ypred_for_auc = sess.run(self.discriminator.ypred_for_auc, feed_dict_discrim)
					ypred = np.array([[item[1], act[1]] for item, act in zip(ypred_for_auc, labels_batch)])
					print ypred
					decoded = decode_captions(captions_batch, self.model.idx_to_word)
					for capt, dec, lab in zip(captions_batch, decoded, labels_batch):
						print capt
						print dec
						print lab
					break
				break

	def rand_shuffle(self, features_batch, all_captions_batch, labels):
		rand_idxs = np.random.permutation(len(all_captions_batch))
		features_batch = features_batch[rand_idxs]
		all_captions_batch = all_captions_batch[rand_idxs]
		labels = labels[rand_idxs]
		return features_batch, all_captions_batch, labels

	def get_generated_captions(self, sess, alphas, betas, sampled_captions, features, n_iters_per_epoch):
		generated_captions = None
		for i in xrange(n_iters_per_epoch):
			features_batch = features[i*self.batch_size:(i+1)*self.batch_size]
			feed_dict_generator = { self.model.features: features_batch}
			_, _, generated_captions_batch = sess.run([alphas, betas, sampled_captions], feed_dict_generator)
			for generated_caption in generated_captions_batch:
				for i in xrange(len(generated_caption) - 1, -1, -1):
					if generated_caption[i] != 2:
						if i < 15:
							generated_caption[i + 1] = 2
						break
					else:
						generated_caption[i] = 0
			if generated_captions is None:
				generated_captions = generated_captions_batch
			generated_captions = np.append(generated_captions, generated_captions_batch, 0)
		return generated_captions

	def train_discrim(self, n_epochs=None):
		# train/val dataset
		n_examples = self.data['captions'].shape[0]
		n_iters_per_epoch = int(np.ceil(float(n_examples)/self.batch_size))
		features = self.data['features']
		captions = self.data['captions']
		image_idxs = self.data['image_idxs']
		val_features = self.val_data['features']
		n_iters_val = int(np.ceil(float(val_features.shape[0])/self.batch_size))

		if n_epochs is None:
			n_epochs = self.n_epochs

		print "The number of epoch: %d" %n_epochs
		print "Data size: %d" %n_examples
		print "Batch size: %d" %self.batch_size
		print "Iterations per epoch: %d" %n_iters_per_epoch

		config = tf.ConfigProto(allow_soft_placement = True)
		config.gpu_options.allow_growth = True
		if self.gpu_list is not None:
			config.gpu_options.visible_device_list = self.gpu_list

		"""
		Training Discrim
		"""
		print "\n\nPre-Training Discriminator ...\n"
		prev_loss = -1
		curr_loss = 0

		# build a graph to sample captions
		alphas, betas, sampled_captions = self.model.build_sampler(max_len=16)    # (N, max_len, L), (N, max_len)

		with tf.Session(config=config) as sess:
			tf.initialize_all_variables().run()

			# Different Loading Paths
			if self.train_new is not None:
				all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
				# print all_vars
				d_vars = set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator"))
				for var in all_vars:
					if var.name in ["beta1_power:0", "beta2_power:0"]:
						print "Adding var:", var
						d_vars.add(var)
				non_d_vars = [item for item in all_vars if item not in d_vars]
				saver = tf.train.Saver(var_list = non_d_vars)
				saver.restore(sess, self.train_new)
				saver = tf.train.Saver(max_to_keep=100)
				saver.save(sess, os.path.join(self.model_path, 'model'), global_step=21)
			elif self.test_model is not None:
				saver = tf.train.Saver()#var_list = non_d_vars)
				saver.restore(sess, self.test_model)
				saver = tf.train.Saver(max_to_keep=100)

			start_t = time.time()

			# Removing <START> token from original captions
			original_captions = captions[:, 1:]

			for e in range(n_epochs):
				# Getting New Training Data
				print "\n\nEpoch:", e
				generated_captions = self.get_generated_captions(sess, alphas, betas, sampled_captions, features, n_iters_per_epoch)
				data_provider = DDataProvider(original_captions, generated_captions)
				all_captions, all_labels = data_provider.get_data()
				for i in range(n_iters_per_epoch):
					captions_batch = all_captions[i*self.batch_size:(i+1)*self.batch_size]
					labels_batch = all_labels[i*self.batch_size:(i+1)*self.batch_size]
					feed_dict_discrim = {
						self.discriminator.input_x: captions_batch,
						self.discriminator.input_y: labels_batch,
						self.discriminator.dropout_keep_prob: self.dis_dropout_keep_prob
					}
					_, new_loss = sess.run([self.discriminator.train_op, self.discriminator.loss], feed_dict_discrim)
					print('Epoch %6d, Step %6d: Loss = %8.3f' % (e, i, new_loss))
					curr_loss += new_loss
				if (e+1) % self.save_every == 0:
					saver.save(sess, os.path.join(self.model_path, 'model'), global_step=22 + e)
					print "model-%s saved." %(e+ 22)
				print "Previous epoch total loss: ", prev_loss
				print "Current epoch total loss: ", curr_loss
				print "Time elapsed: ", time.time() - start_t
				print "\n\n"
				prev_loss = curr_loss
				curr_loss = 0

	def train_gen_init(self):
		n_examples = self.data['captions'].shape[0]
		n_iters_per_epoch = int(np.ceil(float(n_examples)/self.batch_size))
		features = self.data['features']
		captions = self.data['captions']
		image_idxs = self.data['image_idxs']
		val_features = self.val_data['features']
		n_iters_val = int(np.ceil(float(val_features.shape[0])/self.batch_size))

		# build graphs for training model and sampling captions
		loss = self.model.build_model()
		with tf.variable_scope(tf.get_variable_scope()) as scope:
			tf.get_variable_scope().reuse_variables()
			_, _, generated_captions = self.model.build_sampler(max_len=20)

		# train op
		with tf.name_scope('optimizer'):
			optimizer = self.optimizer(learning_rate=self.learning_rate)
			grads = tf.gradients(loss, tf.trainable_variables())
			grads_and_vars = list(zip(grads, tf.trainable_variables()))
			train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)

		# summary op
		tf.summary.scalar('batch_loss', loss)
		for var in tf.trainable_variables():
			tf.summary.histogram(var.op.name, var)
		for grad, var in grads_and_vars:
			tf.summary.histogram(var.op.name+'/gradient', grad)

		summary_op = tf.summary.merge_all()

		print "The number of epoch: %d" %self.n_epochs
		print "Data size: %d" %n_examples
		print "Batch size: %d" %self.batch_size
		print "Iterations per epoch: %d" %n_iters_per_epoch

		config = tf.ConfigProto(allow_soft_placement = True)
		#config.gpu_options.per_process_gpu_memory_fraction=0.9
		config.gpu_options.allow_growth = True
		with tf.Session(config=config) as sess:
			tf.initialize_all_variables().run()
			summary_writer = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())
			saver = tf.train.Saver(max_to_keep=40)

			if self.pretrained_model is not None:
				print "Start training with pretrained Model.."
				saver.restore(sess, self.pretrained_model)

			prev_loss = -1
			curr_loss = 0
			start_t = time.time()

			for e in range(self.n_epochs):
				rand_idxs = np.random.permutation(n_examples)
				captions = captions[rand_idxs]
				image_idxs = image_idxs[rand_idxs]

				for i in range(n_iters_per_epoch):
					captions_batch = captions[i*self.batch_size:(i+1)*self.batch_size]
					image_idxs_batch = image_idxs[i*self.batch_size:(i+1)*self.batch_size]
					features_batch = features[image_idxs_batch]
					feed_dict = {self.model.features: features_batch, self.model.captions: captions_batch}
					_, l = sess.run([train_op, loss], feed_dict)
					curr_loss += l

					# write summary for tensorboard visualization
					if i % 10 == 0:
						summary = sess.run(summary_op, feed_dict)
						summary_writer.add_summary(summary, e*n_iters_per_epoch + i)

					if (i+1) % self.print_every == 0:
						print "\nTrain loss at epoch %d & iteration %d (mini-batch): %.5f" %(e+1, i+1, l)
						ground_truths = captions[image_idxs == image_idxs_batch[0]]
						decoded = decode_captions(ground_truths, self.model.idx_to_word)
						for j, gt in enumerate(decoded):
							print "Ground truth %d: %s" %(j+1, gt)
						gen_caps = sess.run(generated_captions, feed_dict)
						decoded = decode_captions(gen_caps, self.model.idx_to_word)
						print "Generated caption: %s\n" %decoded[0]

				print "Previous epoch loss: ", prev_loss
				print "Current epoch loss: ", curr_loss
				print "Elapsed time: ", time.time() - start_t
				prev_loss = curr_loss
				curr_loss = 0

				# print out BLEU scores and file write
				if self.print_bleu:
					all_gen_cap = np.ndarray((val_features.shape[0], 20))
					for i in range(n_iters_val):
						features_batch = val_features[i*self.batch_size:(i+1)*self.batch_size]
						feed_dict = {self.model.features: features_batch}
						gen_cap = sess.run(generated_captions, feed_dict=feed_dict)
						all_gen_cap[i*self.batch_size:(i+1)*self.batch_size] = gen_cap

					all_decoded = decode_captions(all_gen_cap, self.model.idx_to_word)
					save_pickle(all_decoded, "./data/val/val.candidate.captions.pkl")
					scores = evaluate(data_path='./data', split='val', get_scores=True)
					write_bleu(scores=scores, path=self.model_path, epoch=e)

				# save model's parameters
				if (e+1) % self.save_every == 0:
					saver.save(sess, os.path.join(self.model_path, 'model'), global_step=e+1)
					print "model-%s saved." %(e+1)

	def add_start_to_gen_cap(self, generated_captions):
		return np.insert(generated_captions, 0, 1, axis=1)

	def train_adversarial(self, train_discrim=False, alternate=False):
		# train/val dataset
		n_examples = self.data['captions'].shape[0]
		n_iters_per_epoch = int(np.ceil(float(n_examples)/self.batch_size))
		features = self.data['features']
		captions = self.data['captions']
		image_idxs = self.data['image_idxs']
		val_features = self.val_data['features']

		n_iters_val = int(np.ceil(float(val_features.shape[0])/self.batch_size))

		print "The number of epoch: %d" %self.n_epochs
		print "Data size: %d" %n_examples
		print "Batch size: %d" %self.batch_size
		print "Iterations per epoch: %d" %n_iters_per_epoch

		config = tf.ConfigProto(allow_soft_placement = True)
		config.gpu_options.allow_growth = True
		if self.gpu_list is not None:
			config.gpu_options.visible_device_list = self.gpu_list

		"""
		Training Adversarial
		"""
		print "\n\nStarting Adversarial Training ...\n"
		prev_loss = -1
		curr_loss = 0

		model_num = 0

		# build a graph to sample captions
		with tf.variable_scope(tf.get_variable_scope()):
			loss, g_loss = self.model.build_model()
			tf.get_variable_scope().reuse_variables()
			alphas, betas, sampled_captions = self.model.build_sampler(max_len=16)    # (N, max_len, L), (N, max_len)
			tf.get_variable_scope().reuse_variables()
			self.model.build_rollout(max_len=16)

		with tf.variable_scope(tf.get_variable_scope(), reuse=False):
			optimizer = self.optimizer(learning_rate=self.learning_rate)
			all_trainable_vars = tf.trainable_variables()
			d_trainable_vars = set(tf.trainable_variables(scope="discriminator"))
			trainable_vars = [item for item in all_trainable_vars if item not in d_trainable_vars]
			grads = tf.gradients(g_loss, trainable_vars)
			grads_c = tf.gradients(loss, trainable_vars)
			grads_and_vars = list(zip(grads, trainable_vars))
			grads_and_vars_c = list(zip(grads_c, trainable_vars))
			train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)
			train_op_c = optimizer.apply_gradients(grads_and_vars=grads_and_vars_c)

		with tf.Session(config=config) as sess:
			tf.initialize_all_variables().run()

			# Different Loading Paths
			if self.test_model is not None:
				saver = tf.train.Saver()
				saver.restore(sess, self.test_model)
				saver = tf.train.Saver(max_to_keep=100)
			start_t = time.time()

			for e in range(self.n_epochs):
				rand_idxs = np.random.permutation(n_examples)
				image_idxs = image_idxs[rand_idxs]
				captions = captions[rand_idxs]
				if alternate and e % 2 == 1:
					print "\n\nTraining Generator Using Cross Entropy ...\n"

					for i in range(n_iters_per_epoch):
						captions_batch = captions[i*self.batch_size:(i+1)*self.batch_size]
						image_idxs_batch = image_idxs[i*self.batch_size:(i+1)*self.batch_size]
						features_batch = features[image_idxs_batch]
						feed_dict = {self.model.features: features_batch, self.model.captions: captions_batch}
						_, l = sess.run([train_op_c, loss], feed_dict)
						curr_loss += l
						print "Epoch %6d, Step %6d: Loss = %8.3f" %(e+1, i+1, l)

						if (i+1) % self.print_every == 0:
							ground_truths = captions_batch[:5]
							decoded = decode_captions(ground_truths, self.model.idx_to_word)
							for j, gt in enumerate(decoded):
								print "Ground truth %d: %s" % (j+1, gt)
							features_print = features_batch[:5]
							feed_dict_print = {self.model.features: features_print}
							gen_caps = sess.run(sampled_captions, feed_dict_print)
							decoded = decode_captions(gen_caps, self.model.idx_to_word)
							for j, gc in enumerate(decoded):
								print "Generated caption %d: %s" %(j+1, gc)
				else:
					print "\n\nTraining Generator Using Rewards ...\n"

					for i in range(n_iters_per_epoch):
						image_idxs_batch = image_idxs[i*self.batch_size:(i+1)*self.batch_size]
						features_batch = features[image_idxs_batch]
						captions_batch = captions[i*self.batch_size:(i+1)*self.batch_size]
						feed_dict_generator = { self.model.features: features_batch}
						_, _, generated_captions = sess.run([alphas, betas, sampled_captions], feed_dict_generator)
						rewards = self.model.get_rewards(sess, self.num_rollout, features_batch, generated_captions, self.discriminator, max_length=16)
						# generated_captions = self.add_start_to_gen_cap(generated_captions)
						feed_dict_g_loss = {
							self.model.features: features_batch,
							self.model.captions: captions_batch,
							self.model.rewards: rewards
						}
						_, g_l = sess.run([train_op, g_loss], feed_dict_g_loss)
						curr_loss += g_l
						print "No-Mix Epoch %6d, Step %6d: G-Loss = %8.3f" %(e+1, i+1, g_l)
						if (i+1) % self.print_every == 0:
							ground_truths = captions_batch[:5]
							decoded = decode_captions(ground_truths, self.model.idx_to_word)
							for j, gt in enumerate(decoded):
								print "Ground truth %d: %s" % (j+1, gt)
							features_print = features_batch[:5]
							feed_dict_print = {self.model.features: features_print}
							gen_caps = sess.run(sampled_captions, feed_dict_print)
							decoded = decode_captions(gen_caps, self.model.idx_to_word)
							for j, gc in enumerate(decoded):
								print "Generated caption %d: %s" %(j+1, gc)
							saver.save(sess, os.path.join(self.model_path, 'model-gen'), global_step=300 + model_num)
							print "model-gen-%s saved." %(model_num + 300)
							model_num += 1

				print "Previous epoch loss: ", prev_loss
				print "Current epoch loss: ", curr_loss
				print "Elapsed time: ", time.time() - start_t
				prev_loss = curr_loss
				curr_loss = 0
				if (e+1) % self.save_every == 0:
					saver.save(sess, os.path.join(self.model_path, 'model-gen'), global_step=300 + model_num)
					print "model-gen-%s saved." %(model_num + 300)
					model_num += 1
				if False and train_discrim and (not alternate or e % 2 == 0): # NEED TO FIX THIS. NEED TRAINING HERE UNDER SAME SESSION
					print "\n\nTraining Discriminator ...\n"

					self.train_discrim(n_epochs=1)
					saver.save(sess, os.path.join(self.model_path, 'model-dis'), global_step=300 + e)
					print "model-dis-%s saved." %(e+ 100)

	def get_unique(self, num_rep, items):
		items_taken = []
		for i in xrange(len(items)):
			if i % num_rep == 0:
				items_taken.append(items[i])
		return np.array(items_taken)

	def print_samples(self, data, n_samples = 50):

		'''
		Args:
		- data: dictionary with the following keys:
		- features: Feature vectors of shape (5000, 196, 512)
		- file_names: Image file names of shape (5000, )
		- captions: Captions of shape (24210, 17)
		- image_idxs: Indices for mapping caption to image of shape (24210, )
		- features_to_captions: Mapping feature to captions (5000, 4~5)
		- split: 'train', 'val' or 'test'
		- attention_visualization: If True, visualize attention weights with images for each sampled word. (ipthon notebook)
		- save_sampled_captions: If True, save sampled captions to pkl file for computing BLEU scores.
		'''

		n_examples = data['captions'].shape[0]
		features = data['features']
		captions = data['captions']
		image_idxs = data['image_idxs']
		n_images = image_idxs[-1]
		file_names = data['file_names']
		choices = np.random.choice(n_images, n_samples)
		for i in xrange(0, n_examples, 5):
			image_idx = image_idxs[i]
			if image_idx in choices:
				 image_file = file_names[image_idx]
				 print image_file
				 img = ndimage.imread(image_file)
				 plt.imshow(img)
				 plt.axis('off')
				 plt.show()
				 for j in xrange(5):
					decoded = decode_captions(captions[i + j], self.model.idx_to_word)
					print decoded

	def test(self, data, split='train', attention_visualization=True, save_sampled_captions=False, validation=True):
		'''
		Args:
		- data: dictionary with the following keys:
		- features: Feature vectors of shape (5000, 196, 512)
		- file_names: Image file names of shape (5000, )
		- captions: Captions of shape (24210, 17)
		- image_idxs: Indices for mapping caption to image of shape (24210, )
		- features_to_captions: Mapping feature to captions (5000, 4~5)
		- split: 'train', 'val' or 'test'
		- attention_visualization: If True, visualize attention weights with images for each sampled word. (ipthon notebook)
		- save_sampled_captions: If True, save sampled captions to pkl file for computing BLEU scores.
		'''

		# build a graph to sample captions
		alphas, betas, sampled_captions = self.model.build_sampler(max_len=20)    # (N, max_len, L), (N, max_len)

		num_rep = 5 if validation else 3
		if validation is None:
			num_rep = 1
		config = tf.ConfigProto(allow_soft_placement=True)
		config.gpu_options.allow_growth = True
		n_examples = data['captions'].shape[0]//num_rep
		n_iters_per_epoch = int(np.ceil(float(n_examples)/self.batch_size))
		features = data['features']
		captions = data['captions']
		image_idxs = data['image_idxs']
		file_names = data['file_names']

		captions = self.get_unique(num_rep, captions)
		image_idxs = self.get_unique(num_rep, image_idxs)

		with tf.Session(config=config) as sess:
			saver = tf.train.Saver()
			saver.restore(sess, self.test_model)
			for i in range(n_iters_per_epoch):
				captions_batch = captions[i*self.batch_size:(i+1)*self.batch_size]
				image_idxs_batch = image_idxs[i*self.batch_size:(i+1)*self.batch_size]
				features_batch = features[image_idxs_batch]
				image_files = file_names[image_idxs_batch]
				feed_dict = { self.model.features: features_batch }
				alps, bts, sam_cap = sess.run([alphas, betas, sampled_captions], feed_dict)  # (N, max_len, L), (N, max_len)
				decoded = decode_captions(sam_cap, self.model.idx_to_word)
				decoded_gt = decode_captions(captions_batch, self.model.idx_to_word)
				if attention_visualization:
					for n in range(self.batch_size):
						print "Sampled Caption: %s" %decoded[n]
						print "Ground truth: %s" %decoded_gt[n]

						# Plot original image
						img = ndimage.imread(image_files[n])
						plt.subplot(4, 5, 1)
						plt.imshow(img)
						plt.axis('off')

						# Plot images with attention weights
						words = decoded[n].split(" ")
						for t in range(len(words)):
							if t > 18:
								break
							plt.subplot(4, 5, t+2)
							plt.text(0, 1, '%s(%.2f)'%(words[t], bts[n,t]) , color='black', backgroundcolor='white', fontsize=8)
							plt.imshow(img)
							alp_curr = alps[n,t,:].reshape(14,14)
							alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=16, sigma=20)
							plt.imshow(alp_img, alpha=0.85)
							plt.axis('off')
						plt.show()

					if save_sampled_captions:
						all_sam_cap = np.ndarray((features.shape[0], 20))
						num_iter = int(np.ceil(float(features.shape[0]) / self.batch_size))
						for i in range(num_iter):
							features_batch = features[i*self.batch_size:(i+1)*self.batch_size]
							feed_dict = { self.model.features: features_batch }
							all_sam_cap[i*self.batch_size:(i+1)*self.batch_size] = sess.run(sampled_captions, feed_dict)
						all_decoded = decode_captions(all_sam_cap, self.model.idx_to_word)
						save_pickle(all_decoded, "./data/%s/%s.candidate.captions.pkl" %(split,split))

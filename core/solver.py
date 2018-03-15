import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
import time
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
		# train/val dataset
		n_examples = self.data['captions'].shape[0]
		n_iters_per_epoch = int(np.ceil(float(n_examples)/self.batch_size))
		features = self.data['features']
		captions = self.data['captions']
		image_idxs = self.data['image_idxs']
		val_features = self.val_data['features']
		n_iters_val = int(np.ceil(float(val_features.shape[0])/self.batch_size))

		# build graphs for training model and sampling captions
		# loss = self.model.build_model()
		# with tf.variable_scope(tf.get_variable_scope()) as scope:
		# tf.get_variable_scope().reuse_variables()
		#     _, _, generated_captions = self.model.build_sampler(max_len=20)
		#
		# # train op
		# with tf.name_scope('optimizer'):
		#     optimizer = self.optimizer(learning_rate=self.learning_rate)
		#     grads = tf.gradients(loss, tf.trainable_variables())
		#     grads_and_vars = list(zip(grads, tf.trainable_variables()))
		#     train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)
		#
		# # summary op
		# tf.summary.scalar('batch_loss', loss)
		# for var in tf.trainable_variables():
		#     tf.summary.histogram(var.op.name, var)
		# for grad, var in grads_and_vars:
		#     tf.summary.histogram(var.op.name+'/gradient', grad)

		# summary_op = tf.summary.merge_all()

		print "The number of epoch: %d" %self.n_epochs
		print "Data size: %d" %n_examples
		print "Batch size: %d" %self.batch_size
		print "Iterations per epoch: %d" %n_iters_per_epoch

		config = tf.ConfigProto(allow_soft_placement = True)
		#config.gpu_options.per_process_gpu_memory_fraction=0.9
		config.gpu_options.allow_growth = True

		"""
		Training Discrim : Might need to take the training and sess out of the discrim. pass it in
		"""
		print "\n\nPre-Training Discriminator ...\n"
		prev_loss = -1
		curr_loss = 0
		loss = self.discriminator.build_model()

		# build a graph to sample captions
		alphas, betas, sampled_captions = self.model.build_sampler(max_len=17)    # (N, max_len, L), (N, max_len)

		with tf.Session(config=config) as sess:
			tf.initialize_all_variables().run()
			#all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
			#d_vars = set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="d_lstm"))
			#non_d_vars = [item for item in all_vars if item not in d_vars]
			#print len(non_d_vars)
			saver = tf.train.Saver()#var_list = non_d_vars)
			saver.restore(sess, self.test_model)
			#saver = tf.train.Saver()
			#saver.save(sess, os.path.join(self.model_path, 'model'), global_step=21)
			start_t = time.time()
			for e in range(self.n_epochs):
				rand_idxs = np.random.permutation(n_examples)
				rand_caption_ind = np.random.permutation(n_examples)
				captions = captions[rand_idxs]
				rand_captions = captions[rand_caption_ind]
				image_idxs = image_idxs[rand_idxs]


				for i in range(n_iters_per_epoch):
					captions_batch = captions[i*self.batch_size:(i+1)*self.batch_size]
					wrong_captions_batch = rand_captions[i*self.batch_size:(i+1)*self.batch_size]
					self.fix_wrong_captions(wrong_captions_batch, captions, rand_idxs[i*self.batch_size:(i+1)*self.batch_size], rand_caption_ind[i*self.batch_size:(i+1)*self.batch_size])

					image_idxs_batch = image_idxs[i*self.batch_size:(i+1)*self.batch_size]
					features_batch_single = features[image_idxs_batch]
					features_batch = np.append(features_batch_single, features_batch_single, 0)
					labels = np.append(np.ones((len(captions_batch), 1)), np.zeros((1 * len(wrong_captions_batch), 1)), 0)
					feed_dict = { self.model.features: features_batch_single }
					alps, bts, gen_cap = sess.run([alphas, betas, sampled_captions], feed_dict)

					decoded_1 = decode_captions(captions_batch[:10, 1:], self.model.idx_to_word)
					decoded_2 = decode_captions(gen_cap[:10], self.model.idx_to_word)
					# all_captions_batch = np.append(np.append(captions_batch, wrong_captions_batch, 0), gen_cap, 0)
					all_captions_batch = np.append(captions_batch, wrong_captions_batch, 0)
					#print all_captions_batch.shape, features_batch.shape, labels.shape
					# print captions_batch[0], "\n", gen_cap[0]
					rewards = self.discriminator.get_rewards(features_batch, all_captions_batch)
					print rewards
					break
					# if (i+1) % self.print_every == 0:
					# 	print "\nTrain loss at epoch %d & iteration %d (mini-batch): %.5f" %(e+1, i+1, l)
					# 	ground_truths = captions[image_idxs == image_idxs_batch[0]]
					# 	decoded = decode_captions(ground_truths, self.model.idx_to_word)
					# 	for j, gt in enumerate(decoded):
					# 		print "Ground truth %d: %s" %(j+1, gt)
					# 	gen_caps = sess.run(generated_captions, feed_dict)
					# 	decoded = decode_captions(gen_caps, self.model.idx_to_word)
					# 	print "Generated caption: %s\n" %decoded[0]
				for i, j in zip(decoded_1, decoded_2):
					print i
					print j
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
			if generated_captions is None:
				generated_captions = generated_captions_batch
			generated_captions = np.append(generated_captions, generated_captions_batch, 0)
		return generated_captions

	def train_discrim(self):
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

			for e in range(self.n_epochs):
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

	def test(self, data, split='train', attention_visualization=True, save_sampled_captions=True):
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

		features = data['features']

		# build a graph to sample captions
		alphas, betas, sampled_captions = self.model.build_sampler(max_len=20)    # (N, max_len, L), (N, max_len)

		config = tf.ConfigProto(allow_soft_placement=True)
		config.gpu_options.allow_growth = True
		with tf.Session(config=config) as sess:
			saver = tf.train.Saver()
			saver.restore(sess, self.test_model)
			features_batch, image_files = sample_coco_minibatch(data, self.batch_size)
			feed_dict = { self.model.features: features_batch }
			alps, bts, sam_cap = sess.run([alphas, betas, sampled_captions], feed_dict)  # (N, max_len, L), (N, max_len)
			decoded = decode_captions(sam_cap, self.model.idx_to_word)

			if attention_visualization:
				for n in range(10):
					print "Sampled Caption: %s" %decoded[n]

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

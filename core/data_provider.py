import numpy as np

class DDataProvider(object):
	def __init__(self, original_captions, generated_captions):
		positive_labels = [[0, 1]] * len(original_captions)
		negative_labels = [[1, 0]] * len(generated_captions)
		self.labels = np.concatenate([positive_labels, negative_labels], 0)
		self.captions = np.concatenate([original_captions, generated_captions], 0)
		self.n_examples = len(self.captions)

	def get_data(self):
		rand_idxs = np.random.permutation(self.n_examples)
		self.captions = self.captions[rand_idxs]
        self.labels = self.labels[rand_idxs]
		return self.captions, self.labels

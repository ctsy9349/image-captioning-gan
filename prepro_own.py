from scipy import ndimage
from collections import Counter
from core.vggnet import Vgg19
from core.utils import *

import tensorflow as tf
import numpy as np
import pandas as pd
import hickle
import os
import json

def get_val(filenames):
	# batch size for extracting feature vectors from vggnet.
	batch_size = 2
	# vgg model path
	vgg_model_path = './data/imagenet-vgg-verydeep-19.mat'

	# extract conv5_3 feature vectors
	vggnet = Vgg19(vgg_model_path)
	vggnet.build()
	with tf.Session() as sess:
		tf.initialize_all_variables().run()
		image_batch = np.array(map(lambda x: ndimage.imread(x, mode='RGB'), filenames)).astype(np.float32)
		feats = sess.run(vggnet.features, feed_dict={vggnet.images: image_batch})
	data = {}
	data['filenames'] = filenames
	data['features'] = feats
	return data

if __name__ == "__main__":
	main()

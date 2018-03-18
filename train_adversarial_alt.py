from core.solver_alt import CaptioningSolver
from core.discriminator_conv import Discriminator
from core.model import CaptionGenerator
from core.utils import load_coco_data


def main():
	# load train dataset
	data = load_coco_data(data_path='./data', split='train')
	word_to_idx = data['word_to_idx']

	model = CaptionGenerator(word_to_idx, dim_feature=[196, 512], dim_embed=512,
							dim_hidden=1024, n_time_step=16, prev2out=True,
							ctx2out=True, alpha_c=1.0, selector=True, dropout=True)

	dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16]
	dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100]
	dis_l2_reg_lambda = 0.2

	discrim = Discriminator(sequence_length=16, num_classes=2, vocab_size=len(word_to_idx),
            embedding_size=128, filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)

	solver = CaptioningSolver(model, discrim, data, data, n_epochs=20, batch_size=64, gpu_list="0,1,2", update_rule='adam',
								learning_rate=0.0025, print_every=20, save_every=1, image_path='./image/',
								pretrained_model=None, model_path='model/lstm/', train_new=None,
								test_model='model/lstm/model-42',
								print_bleu=False, log_path='log/', num_rollout=10)

	solver.train_adversarial()

if __name__ == "__main__":
	main()

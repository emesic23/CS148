import torch
from torch import nn
import matplotlib.pyplot as plt
import positional_encoder.utils as utils
from positional_encoder.transformer import Transformer
from positional_encoder.sin_cos_encoder import SinCosTextEncoder, SinCosPosEncoder
from positional_encoder.sin_cos_concat_encoder import SinCosConcatTextEncoder, SinCosConcatPosEncoder
from positional_encoder.index_encoder import IndexTextEncoder, IndexPosEncoder
from positional_encoder.learned_encoder import LearnedTextEncoder, LearnedPosEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run(text_encoder, pos_encoder, exp_name):
	criterion = nn.CrossEntropyLoss()
	n_epochs = 15
	model = Transformer(text_encoder, pos_encoder, n_tokens=n_tokens).to(device)
	train_loss, val_loss, test_loss = utils.train(
		model, 
		train_data, 
		val_data, 
		test_data, 
		n_tokens, 
		n_epochs, 
		criterion, 
		device
	)
	utils.plot_losses(
		title=exp_name, 
		losses={
			'Training Loss': train_loss, 
			'Validation Loss': val_loss
		}
	)
	return model, train_loss, val_loss, test_loss


if __name__ == "__main__":
	print(f'device is {device}\n')

	seed = 0
	torch.manual_seed(seed)

	# load and visualize data
	batch_size = 64
	train_data, val_data, test_data, vocab = utils.get_data(batch_size)

	print(f'Training Data size: {train_data.size()}')
	print(f'Validation Data size: {val_data.size()}')
	print(f'Test Data size: {test_data.size()}\n')

	idxs, batch = list(range(35, 50)), 0
	tokens = train_data[idxs, batch]
	words = [vocab.lookup_token(token) for token in tokens]
	n_tokens = len(vocab)
	print(f'number of tokens: {len(vocab)}')

	print(f'example tokens:\n{[t.item() for t in tokens]}')
	print(f'words:\n{words}\n')

	print('Training SinCos Encoder')

	# sin and cos positional encoders
	sincos_model, sincos_train_loss, sincos_val_loss , _ = run(
		SinCosTextEncoder,
		SinCosPosEncoder,
		'SinCos Encoding'
	)

	print('Generating Positional Encoder Visualization')

	# visualize sin cosine positional encoding
	fig, ax = plt.subplots(figsize=(8, 4))
	embed = ax.imshow(sincos_model.pos_encoder.positional_encoding[:100, 0, :].to('cpu'), aspect='auto')
	fig.colorbar(embed, ax=ax)
	plt.savefig(f'part_2_plots/p2_sincos_encoding_visualization.png', dpi=300)

	fig, axes = plt.subplots(2, figsize=(10, 6))
	sin = axes[0].imshow(sincos_model.pos_encoder.positional_encoding[:100, 0, 0::2].to('cpu'), aspect='auto')
	cos = axes[1].imshow(sincos_model.pos_encoder.positional_encoding[:100, 0, 1::2].to('cpu'), aspect='auto')
	fig.colorbar(sin, ax=axes[0])
	fig.colorbar(cos, ax=axes[1])
	plt.savefig(f'part_2_plots/p2_sincos_encoding_visualization_odd_even.png', dpi=300)

	print('Training SinCosConcat Encoder')

	# sin and cos concatenate positional encoders
	sincoscat_model, sincoscat_train_loss, sincoscat_val_loss , _ = run(
		SinCosConcatTextEncoder,
		SinCosConcatPosEncoder,
		'SinCos Concat Encoding'
	)

	print('Training Idx Encoder')

	# index positional encoders
	index_model, index_train_loss, index_val_loss , _ = run(
		IndexTextEncoder,
		IndexPosEncoder,
		'Index Encoding'
	)

	print('Training Learned Encoder')

	# learned positional encoders
	learned_model, learned_train_loss, learned_val_loss , _ = run(
		LearnedTextEncoder,
		LearnedPosEncoder,
		'Learned Positional Encoding'
	)

	# compare positional encodings
	utils.plot_losses(
		title='Positional Encoding Comparison',
		losses={
			'SinCos': (sincos_train_loss, sincos_val_loss),
			'SinCosCat': (sincoscat_train_loss, sincoscat_val_loss),
			'Idx': (index_train_loss, index_val_loss),
			'Learned': (learned_train_loss, learned_val_loss)
		}
	)
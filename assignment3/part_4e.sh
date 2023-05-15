##! /bin/bash

# pretrain on the wiki dataset
python nanogpt/run.py pretrain \
	--char_corruption False \
	--writing_params_path nanogpt/checkpoints/p4e_pretrain.pt

# finetune on the names dataset
python nanogpt/run.py finetune \
	--finetune_corpus_path nanogpt/data/birth_places_train.tsv \
	--reading_params_path nanogpt/checkpoints/p4e_pretrain.pt \
    --writing_params_path nanogpt/checkpoints/p4e_finetune.pt

# evaluate on the test set and save predictions
python nanogpt/run.py evaluate \
	--evaluate_corpus_path nanogpt/data/birth_places_test.tsv \
	--reading_params_path nanogpt/checkpoints/p4e_finetune.pt \
    --outputs_path nanogpt/predictions/p4e_finetune.txt

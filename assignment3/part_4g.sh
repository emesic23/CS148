##! /bin/bash

# pretrain on the wiki dataset with character corruption
python nanogpt/run.py pretrain \
	--char_corruption True \
	--writing_params_path nanogpt/checkpoints/p4g_pretrain.pt

# finetune on the names dataset
python nanogpt/run.py finetune \
	--finetune_corpus_path nanogpt/data/birth_places_train.tsv \
	--reading_params_path nanogpt/checkpoints/p4g_pretrain.pt \
    --writing_params_path nanogpt/checkpoints/p4g_finetune.pt

# evaluate on the test set and save predictions
python nanogpt/run.py evaluate \
	--evaluate_corpus_path nanogpt/data/birth_places_test.tsv \
	--reading_params_path nanogpt/checkpoints/p4g_finetune.pt \
    --outputs_path nanogpt/predictions/p4g_finetune.txt

##! /bin/bash

# train on the names dataset
python nanogpt/run.py finetune \
    --finetune_corpus_path nanogpt/data/birth_places_train.tsv \
    --writing_params_path nanogpt/checkpoints/p4c_finetune_without_pretrain.pt

# evaluate on the test set and save predictions
python nanogpt/run.py evaluate \
    --evaluate_corpus_path nanogpt/data/birth_places_test.tsv \
    --reading_params_path nanogpt/checkpoints/p4c_finetune_without_pretrain.pt \
    --outputs_path nanogpt/predictions/p4c_finetune_without_pretrain.txt

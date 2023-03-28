#!/bin/bash

base_dir=/home/azureuser/cloudfiles/code/Users/jimdilkes/bert-prune
models_dir=$base_dir/models

sparsity=30
out_model=$models_dir/bert-prune-$sparsity
in_model=$models_dir/uncased-bert-100-pretrain-20k-base

bert_dir=$base_dir/bert
train=$base_dir/data/pretraining/train
dev=$base_dir/data/pretraining/dev
orig_model=$models_dir/uncased-bert-prunable

sparsity_args="--pruning_hparams=initial_sparsity=0,target_sparsity=.$sparsity,"
      sparsity_args+="sparsity_function_end_step=5000,end_pruning_step=-1"

params="--train_batch_size 32
--max_seq_length 128
--max_predictions_per_seq 20
--num_warmup_steps 10
--save_checkpoints_steps 2000
--keep_checkpoint_max 40
--learning_rate 2e-5
--output_dir $out_model
--bert_config_file $orig_model/bert_config.json
--init_checkpoint $in_model/model.ckpt"

# out_dir="--output_dir $out_model/step_$step"
python $bert_dir/run_pretraining.py --do_train=True --num_train_steps=20000 --input_file=$train/* $params $sparsity_args
python $bert_dir/run_pretraining.py --do_eval=True --max_eval_steps=2000 --input_file=$dev/* $params $sparsity_args

# for step in {1..1}; do 
#     out_dir="--output_dir $out_model/step_$step"
#     python $bert_dir/run_pretraining.py --do_train=True --num_train_steps=200 --input_file=$train/* $params $out_dir $sparsity_args
#     python $bert_dir/run_pretraining.py --do_eval=True --max_eval_steps=20 --input_file=$dev/* $params $out_dir $sparsity_args
# done

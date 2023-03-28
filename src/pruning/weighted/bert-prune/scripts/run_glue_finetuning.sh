
base_dir=/home/azureuser/cloudfiles/code/Users/jimdilkes/bert-prune
bert_dir=$base_dir/bert
in_model=$base_dir/uncased-bert-100-pretrain-FT
orig_model=$base_dir/uncased-bert-prunable
out_log_dir=$out_model/logs
out_model=$in_model-3

glue_data_dir=$base_dir/data/glue_data


for glue_task in CoLA SST-2 QNLI QQP MNLI
do
    params="--task_name $glue_task
    --data_dir $glue_data_dir/$glue_task
    --init_checkpoint $in_model
    --output_dir $out_model 
    --bert_config_file $orig_model/bert_config.json
    --vocab_file $orig_model/vocab.txt
    --train_batch_size 32
    --max_seq_length 128
    --keep_checkpoint_max 1
    --learning_rate 2e-5"

    for epoch in $(seq 1 2)
    do
        # echo python $bert_dir/run_classifier.py --do_train=True --num_train_epochs=$epoch $params
        python $bert_dir/run_classifier.py --do_train=asd --num_train_epochs=$epoch $params
        # python $bert_dir/run_classifier.py --do_eval=True $params
    done
done
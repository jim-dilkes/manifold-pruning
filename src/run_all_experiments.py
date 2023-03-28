import argparse
import os

DATASET_BASE = 'data/experiments'
FEATURES_BASE = 'data/features'
MASKS_BASE = 'models/masks/bert-base-uncased-squad2/squad_v2'
RESULTS_BASE = 'results'
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# parse args
parser = argparse.ArgumentParser(description='Run all experiments for a given pruning metric.')
parser.add_argument('--pruning_metric', type=str, default=None,
                    choices=['random', 'mac', 'latency'],
                    help='Input a supported pruning metric')
args = parser.parse_args()

# define which experiments to run
pruned_percentages = range(0, 100, 10)
experiment_ids = range(1,6)

# extract features and evaluate mftma for each experiment
for id in experiment_ids:
    dataset_file = os.path.join(parent_dir, DATASET_BASE, 'FINAL_Q.pkl')
    tag_file = os.path.join(parent_dir, DATASET_BASE, f'relevant_pos_tags_{id}.txt')
    sample = os.path.join(parent_dir, DATASET_BASE, f'sample_seed_{id}.pkl')

    for perc in pruned_percentages:
        if perc == 0:
            feature_dir = os.path.join(parent_dir, FEATURES_BASE, f'train_{id}', '0Pruned')
            # example: data/features/train_1/0Pruned
            mftma_analysis_dir = os.path.join(parent_dir, RESULTS_BASE, f'train_{id}', '0Pruned')
            # example: results/masked/train_1/0Pruned
            
            os.system(f'python feature_extract.py \
                    --dataset_file {dataset_file} \
                    --tag_file {tag_file} \
                    --sample {sample} \
                    --feature_dir {feature_dir}')
            
            os.system(f'python mftma_analysis.py \
                    --feature_dir {feature_dir} \
                    --mftma_analysis_dir {mftma_analysis_dir}\
                    --n_t 10 \
                    --n_reps 1 \
                    --num_layers 2')

        else:
            feature_dir = os.path.join(parent_dir, FEATURES_BASE, f'train_{id}', args.pruning_metric, f'{perc}Pruned')
            # example: data/features/train_1/random/10Pruned
            mftma_analysis_dir = os.path.join(parent_dir, RESULTS_BASE, f'train_{id}', args.pruning_metric, f'{perc}Pruned')
            # example: results/train_1/random/10Pruned
            masks_dir = os.path.join(parent_dir, MASKS_BASE, args.pruning_metric, f'{round((100 - perc) / 100, 1)}/seed_0')
            # example: models/masks/bert-base-uncased-squad2/squad_v2/random/0.1/seed_0
            
            os.system(f'python feature_extract.py \
                    --dataset_file {dataset_file} \
                    --tag_file {tag_file} \
                    --sample {sample} \
                    --pruning_metric {args.pruning_metric} \
                    --pruned_percentage {perc} \
                    --masks_dir {masks_dir} \
                    --feature_dir {feature_dir}')

            os.system(f'python mftma_analysis.py \
                    --pruning_metric {args.pruning_metric} \
                    --pruned_percentage {perc} \
                    --feature_dir {feature_dir} \
                    --mftma_analysis_dir {mftma_analysis_dir}\
                    --n_t 10 \
                    --n_reps 1 \
                    --num_layers 2')

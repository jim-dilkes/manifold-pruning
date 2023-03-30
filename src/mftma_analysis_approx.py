import os
from time import time
import argparse
import pickle as pkl
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from mftma.manifold_analysis_correlation import manifold_analysis_corr_approx

def main():
    parser = argparse.ArgumentParser(description='MFTMA analysis over layers.')

    # Input
    parser.add_argument('--feature_dir', type=str, default='data/features',
                        help='Input feature data directory.')
    parser.add_argument('--pruning_metric', type=str,
                        choices=['random', 'mac', 'latency'],
                        help='Input a supported pruning metric')
    parser.add_argument('--pruned_percentage', type=int,
                        default=0,
                        help='Percentage of the model that has been pruned.')

    #Output
    parser.add_argument('--mftma_analysis_dir', type=str, default='results/mftma-analysis-approx',
                        help='Location to output MFTMA analysis directory.')

    parser.add_argument('--num_layers', type=int, default=12, help='Number of hidden layers.')

    # MFTMA parameters
    parser.add_argument('--kappa', type=float, default=1e-8, help='Margin size to use in the '
                                                                'analysis (kappa > 0).')
    parser.add_argument('--n_t', type=int, default=1000, help='Number of gaussian vectors to sample '
                                                            'per manifold.')

    parser.add_argument('--n_reps', type=int, default=10, help='Number of repetitions.')

    args = parser.parse_args()
    print(args)

    for layer in range(1,args.num_layers+1):
        start_time = time()
        print(f'============ layer {layer} ============')
        class_encodings = pkl.load(
            open(os.path.join(args.feature_dir, f'{str(layer)}.pkl'), 'rb+')
        )
        print(len(class_encodings))
        print(class_encodings[0].shape)

        # class_encodings = rearrange(np.array(class_encodings), 'm d n -> m n d')
        # class_encodings = F.normalize(class_encodings, dim=-1)
        r, d, a = manifold_analysis_corr_approx(class_encodings)
        mftma_analysis_data = {'a': a, 'r': r, 'd': d, 'r0': 1, 'K': 1}

        os.makedirs(args.mftma_analysis_dir, exist_ok=True)
        pkl.dump(
            mftma_analysis_data,
            open(
                os.path.join(args.mftma_analysis_dir, f'{str(layer)}.pkl'), 'wb+'
            ),
        )

if __name__ == "__main__":
    main()

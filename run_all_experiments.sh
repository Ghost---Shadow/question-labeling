#!/bin/bash
set -euo pipefail

echo $HOSTNAME

# Batch size sweeps
source experiments/batch_size_sweep/batch_size_sweep_experiments.sh

# Loss fn sweep
source experiments/loss_fn_sweep/loss_fn_sweep_experiments.sh

# Cross dataset generalization
source experiments/cross_dataset_generalization/cross_dataset_generalization_experiments.sh

# QD ablation
source experiments/qd_ablation.sh

# non-iterative strategy
# python ./src/train.py --config=experiments/gpt35_mpnet_ni_triplet.yaml --debug
python ./src/train.py --config=experiments/gpt35_mpnet_ni_triplet.yaml

# Final model, train on 2wikihop
# python ./src/train.py --config=experiments/gpt35_mpnet_triplet_wiki.yaml --debug
python ./src/train.py --config=experiments/gpt35_mpnet_triplet_wiki.yaml

source devops/stop_current_gcp_instance.sh

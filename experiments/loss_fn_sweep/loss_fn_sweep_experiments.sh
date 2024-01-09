#!/bin/bash
set -euo pipefail

echo $HOSTNAME

# python ./src/train.py --config=experiments/loss_fn_sweep/gpt35_mpnet_kldiv.yaml --debug
python ./src/train.py --config=experiments/loss_fn_sweep/gpt35_mpnet_kldiv.yaml

# python ./src/train.py --config=experiments/loss_fn_sweep/gpt35_mpnet_mse.yaml --debug
python ./src/train.py --config=experiments/loss_fn_sweep/gpt35_mpnet_mse.yaml

# python ./src/train.py --config=experiments/loss_fn_sweep/gpt35_mpnet_triplet.yaml --debug
python ./src/train.py --config=experiments/loss_fn_sweep/gpt35_mpnet_triplet.yaml

source devops/stop_current_gcp_instance.sh

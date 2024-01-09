#!/bin/bash
set -euo pipefail

echo $HOSTNAME

# python ./src/train.py --config=experiments/batch_size_sweep_kldiv/gpt35_mpnet_kldiv_batch_2.yaml --debug
python ./src/train.py --config=experiments/batch_size_sweep_kldiv/gpt35_mpnet_kldiv_batch_2.yaml

# python ./src/train.py --config=experiments/batch_size_sweep_kldiv/gpt35_mpnet_kldiv_batch_16.yaml --debug
python ./src/train.py --config=experiments/batch_size_sweep_kldiv/gpt35_mpnet_kldiv_batch_16.yaml

# python ./src/train.py --config=experiments/batch_size_sweep_kldiv/gpt35_mpnet_kldiv_batch_32.yaml --debug
python ./src/train.py --config=experiments/batch_size_sweep_kldiv/gpt35_mpnet_kldiv_batch_32.yaml

# python ./src/train.py --config=experiments/batch_size_sweep_kldiv/gpt35_mpnet_kldiv_batch_48.yaml --debug
python ./src/train.py --config=experiments/batch_size_sweep_kldiv/gpt35_mpnet_kldiv_batch_48.yaml

# python ./src/train.py --config=experiments/batch_size_sweep_kldiv/gpt35_mpnet_kldiv_batch_64.yaml --debug
python ./src/train.py --config=experiments/batch_size_sweep_kldiv/gpt35_mpnet_kldiv_batch_64.yaml

source devops/stop_current_gcp_instance.sh

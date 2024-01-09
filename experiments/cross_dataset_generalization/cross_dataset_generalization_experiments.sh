#!/bin/bash
set -euo pipefail

echo $HOSTNAME

# python ./src/train.py --config=experiments/cross_dataset_generalization/gpt35_mpnet_kldiv_hotpot.yaml --debug
python ./src/train.py --config=experiments/cross_dataset_generalization/gpt35_mpnet_kldiv_hotpot.yaml

# python ./src/train.py --config=experiments/cross_dataset_generalization/gpt35_mpnet_kldiv_wiki.yaml --debug
python ./src/train.py --config=experiments/cross_dataset_generalization/gpt35_mpnet_kldiv_wiki.yaml

source devops/stop_current_gcp_instance.sh

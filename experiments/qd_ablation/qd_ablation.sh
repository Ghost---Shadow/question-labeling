#!/bin/bash
set -euo pipefail

echo $HOSTNAME $ZONE

# python ./src/train.py --config=experiments/qd_ablation/gpt35_mpnet_triplet_only_d.yaml --debug
python ./src/train.py --config=experiments/qd_ablation/gpt35_mpnet_triplet_only_d.yaml

# python ./src/train.py --config=experiments/qd_ablation/gpt35_mpnet_triplet_only_q.yaml --debug
python ./src/train.py --config=experiments/qd_ablation/gpt35_mpnet_triplet_only_q.yaml

source devops/stop_current_gcp_instance.sh

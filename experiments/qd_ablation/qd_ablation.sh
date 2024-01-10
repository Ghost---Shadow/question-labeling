#!/bin/bash
set -euo pipefail

echo $HOSTNAME

# python ./src/train.py --config=experiments/qd_ablation/gpt35_mpnet_kldiv_only_d.yaml --debug
python ./src/train.py --config=experiments/qd_ablation/gpt35_mpnet_kldiv_only_d.yaml

# python ./src/train.py --config=experiments/qd_ablation/gpt35_mpnet_kldiv_only_q.yaml --debug
python ./src/train.py --config=experiments/qd_ablation/gpt35_mpnet_kldiv_only_q.yaml

# python ./src/train.py --config=experiments/qd_ablation/gpt35_mpnet_kldiv_qd.yaml --debug
python ./src/train.py --config=experiments/qd_ablation/gpt35_mpnet_kldiv_qd.yaml

# python ./src/train.py --config=experiments/qd_ablation/gpt35_mpnet_ni_kldiv.yaml --debug
python ./src/train.py --config=experiments/qd_ablation/gpt35_mpnet_ni_kldiv.yaml

source devops/stop_current_gcp_instance.sh

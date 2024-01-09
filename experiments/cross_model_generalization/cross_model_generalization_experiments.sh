#!/bin/bash
set -euo pipefail

echo $HOSTNAME

# python ./src/train.py --config=experiments/cross_model_generalization/gpt35_beamr_kldiv.yaml --debug
python ./src/train.py --config=experiments/cross_model_generalization/gpt35_beamr_kldiv.yaml

# # python ./src/train.py --config=experiments/cross_model_generalization/gpt35_dbertav3_kldiv.yaml --debug
# python ./src/train.py --config=experiments/cross_model_generalization/gpt35_dbertav3_kldiv.yaml

source devops/stop_current_gcp_instance.sh

#!/bin/bash
set -euo pipefail

echo $HOSTNAME

# Start both Python scripts in the background
python ./src/train.py --config=experiments/qd_ablation/gpt35_mpnet_kldiv_only_q.yaml &
python ./src/train.py --config=experiments/qd_ablation/gpt35_mpnet_kldiv_qd.yaml &

# Wait for both Python scripts to finish
wait

# The script will only reach this line once both Python scripts have completed
source devops/stop_current_gcp_instance.sh

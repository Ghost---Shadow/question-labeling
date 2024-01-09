# Batch size sweeps
source experiments/batch_size_sweep/batch_size_sweep_experiments.sh

# Loss fn sweep
source experiments/loss_fn_sweep/loss_fn_sweep_experiments.sh

# QD ablation
source experiments/qd_ablation.sh

# non-iterative strategy
# python ./src/train.py --config=experiments/gpt35_mpnet_ni_triplet.yaml --debug
python ./src/train.py --config=experiments/gpt35_mpnet_ni_triplet.yaml

# Final model, train on 2wikihop
# python ./src/train.py --config=experiments/gpt35_mpnet_triplet_wiki.yaml --debug
python ./src/train.py --config=experiments/gpt35_mpnet_triplet_wiki.yaml

gcloud compute instances stop q-labeling-2 --zone=us-central1-a --project=angular-unison-350808

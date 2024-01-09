# python ./src/train.py --config=experiments/qd_ablation/gpt35_mpnet_triplet_only_d.yaml --debug
python ./src/train.py --config=experiments/qd_ablation/gpt35_mpnet_triplet_only_d.yaml

# python ./src/train.py --config=experiments/qd_ablation/gpt35_mpnet_triplet_only_q.yaml --debug
python ./src/train.py --config=experiments/qd_ablation/gpt35_mpnet_triplet_only_q.yaml

gcloud compute instances stop q-labeling-1 --zone=us-central1-a --project=angular-unison-350808

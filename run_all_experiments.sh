# python ./src/train.py --config=experiments/gpt35_mpnet_triplet_only_d.yaml --debug
python ./src/train.py --config=experiments/gpt35_mpnet_triplet_only_d.yaml

# python ./src/train.py --config=experiments/gpt35_mpnet_triplet_only_q.yaml --debug
python ./src/train.py --config=experiments/gpt35_mpnet_triplet_only_q.yaml

# python ./src/train.py --config=experiments/gpt35_mpnet_ni_triplet.yaml --debug
python ./src/train.py --config=experiments/gpt35_mpnet_ni_triplet.yaml

# python ./src/train.py --config=experiments/gpt35_mpnet_triplet.yaml --debug
python ./src/train.py --config=experiments/gpt35_mpnet_triplet.yaml

# python ./src/train.py --config=experiments/gpt35_mpnet_triplet_wiki.yaml --debug
python ./src/train.py --config=experiments/gpt35_mpnet_triplet_wiki.yaml

# python ./src/train.py --config=experiments/gpt35_mpnet_kldiv.yaml --debug
python ./src/train.py --config=experiments/gpt35_mpnet_kldiv.yaml

# python ./src/train.py --config=experiments/gpt35_mpnet_mse.yaml --debug
python ./src/train.py --config=experiments/gpt35_mpnet_mse.yaml

gcloud compute instances stop q-labeling-1 --zone=us-central1-a --project=angular-unison-350808
# python devops/stop_instance.py

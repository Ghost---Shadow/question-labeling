# python ./src/train.py --config=experiments/batch_size_sweep/gpt35_mpnet_triplet_batch_2.yaml --debug
python ./src/train.py --config=experiments/batch_size_sweep/gpt35_mpnet_triplet_batch_2.yaml

# python ./src/train.py --config=experiments/batch_size_sweep/gpt35_mpnet_triplet_batch_4.yaml --debug
python ./src/train.py --config=experiments/batch_size_sweep/gpt35_mpnet_triplet_batch_4.yaml

# python ./src/train.py --config=experiments/batch_size_sweep/gpt35_mpnet_triplet_batch_16.yaml --debug
python ./src/train.py --config=experiments/batch_size_sweep/gpt35_mpnet_triplet_batch_16.yaml

# python ./src/train.py --config=experiments/batch_size_sweep/gpt35_mpnet_triplet_batch_256.yaml --debug
python ./src/train.py --config=experiments/batch_size_sweep/gpt35_mpnet_triplet_batch_256.yaml

gcloud compute instances stop q-labeling-1 --zone=us-central1-a --project=angular-unison-350808

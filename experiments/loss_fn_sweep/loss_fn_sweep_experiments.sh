# python ./src/train.py --config=experiments/loss_fn_sweep/gpt35_mpnet_kldiv.yaml --debug
python ./src/train.py --config=experiments/loss_fn_sweep/gpt35_mpnet_kldiv.yaml

# python ./src/train.py --config=experiments/loss_fn_sweep/gpt35_mpnet_mse.yaml --debug
python ./src/train.py --config=experiments/loss_fn_sweep/gpt35_mpnet_mse.yaml

# python ./src/train.py --config=experiments/loss_fn_sweep/gpt35_mpnet_triplet.yaml --debug
python ./src/train.py --config=experiments/loss_fn_sweep/gpt35_mpnet_triplet.yaml

gcloud compute instances stop q-labeling-2 --zone=us-central1-a --project=angular-unison-350808

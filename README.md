# Question Labeling

## Installation

```sh
conda create -n qlabeling python=3.9 -y
conda activate qlabeling
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118
```

## Linting

```sh
black .
```

## Tests

```sh
python -m unittest discover -s src -p "*_test.py"
```

## Experiments

Make sure `./.env` is populated with correct keys

```sh
python ./src/train.py --config=experiments/gpt35_mpnet_avg.yaml --debug
python ./src/train.py --config=experiments/gpt35_mpnet_avg.yaml

python ./src/train.py --config=experiments/gpt35_mpnet_smi_avg.yaml --debug
python ./src/train.py --config=experiments/gpt35_mpnet_smi_avg.yaml

python ./src/train.py --config=experiments/mpnet_avg.yaml --debug
python ./src/train.py --config=experiments/mpnet_avg.yaml

python ./src/train.py --config=experiments/gpt35_mpnet_ni_triplet.yaml --debug
python ./src/train.py --config=experiments/gpt35_mpnet_ni_triplet.yaml
```

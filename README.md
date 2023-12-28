# Question Labeling

## Installation

```sh
source ./devops/install.sh
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
python ./src/train.py --config=experiments/gpt35_mpnet_ni_triplet.yaml --debug
python ./src/train.py --config=experiments/gpt35_mpnet_ni_triplet.yaml

python ./src/train.py --config=experiments/gpt35_mpnet_triplet.yaml --debug
python ./src/train.py --config=experiments/gpt35_mpnet_triplet.yaml
```

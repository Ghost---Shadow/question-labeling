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

```sh
python ./src/train.py --config=experiments/gpt35_mpnet_avg.yaml --debug
python ./src/train.py --config=experiments/gpt35_mpnet_avg.yaml
```

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
sh experiments/batch_size_sweep/batch_size_sweep_experiments.sh
sh run_all_experiments.sh
```

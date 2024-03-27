# Early hyperacute stroke dataset

## Installation

Build docker image:
```commandline
./build.sh
```

## Usage

First, run container:

```commandline
./run.sh %--gpus docker option value%
```

Then all commands are executed in the terminal of the running container. If you run training, first run MLFlow not in 
container:

```commandline
./run_mlflow.sh
```

Training:

```commandline
python3 early_hyperacute_stroke_dataset/scripts/train.py settings/baseline.yaml
```

Inference:

```commandline
python3 early_hyperacute_stroke_dataset/scripts/inference.py [-h] --model MODEL --settings SETTINGS \
--dataset_metadata DATASET_METADATA --data DATA --output OUTPUT [--device_type {cpu,cuda}] [--device_num DEVICE_NUM]
```

Result analysis:

```commandline
python3 early_hyperacute_stroke_dataset/scripts/analysis.py [-h] --predict PREDICT --reference REFERENCE --output OUTPUT
```

Result visualisation:

```commandline
python3 early_hyperacute_stroke_dataset/scripts/vis_predict.py [-h] --predict PREDICT --reference REFERENCE \
--output OUTPUT [--show_reference]
```

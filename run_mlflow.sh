#!/usr/bin/env bash

path=$(pwd)

uid=$(id -u)
gid=$(id -g)

docker run -it --rm -u "$uid":"$gid" --net host \
 --volume "$path":/app:rw early_hyperacute_stroke_dataset:latest mlflow server \
 --backend-store-uri /app/data/mlflow  --default-artifact-root /app/data/mlflow  --host 0.0.0.0

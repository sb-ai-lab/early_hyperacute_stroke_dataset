#!/usr/bin/env bash
path=$(pwd)

uid=$(id -u)
gid=$(id -g)

shm_size="16G"

if [[ $1 ]]; then
  gpus=$1
else
  gpus="all"
fi

docker run -it --rm --gpus $gpus -u "$uid":"$gid" --net host --shm-size "$shm_size" \
  --volume "$path":/app:rw early_hyperacute_stroke_dataset /bin/bash

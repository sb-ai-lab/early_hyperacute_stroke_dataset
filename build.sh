#!/usr/bin/env bash

docker build --no-cache --network host -t early_hyperacute_stroke_dataset -f Dockerfile .

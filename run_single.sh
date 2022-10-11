#!/bin/bash

dataset=$1
model=$2
confidence=$3
label=$4

data_file=data/$dataset/preds_$model.json
if [[ $label == "soft" ]]
then
    name=${dataset}_${model}_${confidence}_soft
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/emnlp22/train.py \
        -t \
        -sl \
        -bs 64 \
        -df $data_file \
        -c $confidence \
        -n $name;
else
    name=${dataset}_${model}_${confidence}
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/emnlp22/train.py \
        -t \
        -bs 64 \
        -df $data_file \
        -c $confidence \
        -n $name;
fi
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/emnlp22/benchmark.py \
        -w models/$name.bin \
        -df $data_file \
        -bs 1024 \
        -n $name

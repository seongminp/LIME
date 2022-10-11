#! /bin/bash

for dataset in agnews yahoo dbpedia tweet clickbait
#for dataset in tweet
do
    #for model in entailment nsp rnsp qa xclass lotclass
    for model in entailment
    do 
        for confidence in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
        do
            data_file=data/$dataset/preds_$model.json
            for label_type in normal soft
            do
                if [[ $label_type == "soft" ]]
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
            done
        done
    done
    name=${dataset}_supervised
    test_file=data/$dataset/data.json
    train_file=data/$dataset/data_train.json
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/emnlp22/train.py \
        -t \
        -bs 64 \
        -df $train_file \
        -c 0 \
        -n $name;
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/emnlp22/benchmark.py \
        -w models/$name.bin \
        -df $test_file \
        -bs 1024 \
        -n $name
done

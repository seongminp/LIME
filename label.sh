model=$1
device=$2

python scripts/emnlp22/create_pseudo_labels.py -d data/agnews/data.json -o data/agnews/preds_$model.json -v topic -m $model -dv $device &&
python scripts/emnlp22/create_pseudo_labels.py -d data/yahoo/data.json -o data/yahoo/preds_$model.json -v topic -m $model -dv $device &&
python scripts/emnlp22/create_pseudo_labels.py -d data/dbpedia/data.json -o data/dbpedia/preds_$model.json -v topic -m $model -dv $device &&
python scripts/emnlp22/create_pseudo_labels.py -d data/tweet/data.json -o data/tweet/preds_$model.json -v emotion -m $model -dv $device &&
python scripts/emnlp22/create_pseudo_labels.py -d data/clickbait/data.json -o data/clickbait/preds_$model.json -v clickbait -m $model -dv $device

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import confusion_matrix, f1_score
from tqdm import tqdm
from train import predict
from transformers import BertTokenizerFast


def benchmark(data, weights, batch_size, language):

    classes = [c[:9] for c in data["classes"]]
    preds = predict(data, weights, batch_size, language)
    labels = [sample["label"] for sample in data["data"]]

    macro_f1 = f1_score(labels, preds, average="macro")
    micro_f1 = f1_score(labels, preds, average="micro")
    matrix = confusion_matrix(labels, preds)

    experiment_name = Path(weights).stem
    print("🔥🔥🔥🔥🔥🔥🔥🔥Final Stats🔥🔥🔥🔥🔥🔥🔥🔥🔥")
    print(f"[{experiment_name}] Micro f1: {micro_f1}, Macro f1: {macro_f1}")
    print(f"[{experiment_name}] Confusion matrix:")
    name_string = "".join(f"{name:10s}" for name in classes)
    print(f"{' '*10}{name_string}")
    for name, row in zip(classes, matrix):
        row_string = "".join(f"{str(v):10s}" for v in row)
        print(f"{name:10s}{row_string}")
    print("🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥")
    return macro_f1, micro_f1


if __name__ == "__main__":

    # Example:
    #
    # python scripts/emnlp22/benchmark.py \
    #     -w classifier.bin \
    #     -df data/AGNews/data.json \
    #     -bs 1024

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", help="Model weights")
    parser.add_argument("-df", "--data-file", help="File with data")
    # parser.add_argument("-d", "--device", help="GPU id", default=2)
    parser.add_argument("-bs", "--batch-size", help="Batch size", default=1, type=int)
    parser.add_argument("-l", "--language", help="en or kr", default="en")
    parser.add_argument("-n", "--name", help="Name of experiment", default="result")
    args = parser.parse_args()

    with open(args.data_file) as rf:
        data = json.load(rf)

    macro_f1, micro_f1 = benchmark(
        data,
        args.weights,
        args.batch_size,
        args.language,
    )

    with open(f"results/{args.name}.txt", "w") as wf:
        wf.write(f"Macro F1,Micro F1\n{macro_f1},{micro_f1}")

import argparse
import json
from pathlib import Path

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-directory", help="Data directory")
    args = parser.parse_args()

    datasets = ["agnews", "yahoo", "dbpedia", "clickbait"]

    for dataset in datasets:
        with open(f"{Path(args.data_directory)}/{dataset}/preds_entailment.json") as rf:
            data = json.load(rf)

        word_count = 0
        for sample in data["data"]:
            words = sample["text"].split()
            word_count += len(words)
        print(f"{dataset}: {word_count / len(data['data'])}")
        

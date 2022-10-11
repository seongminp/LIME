import argparse
import json

from sklearn.metrics import confusion_matrix, f1_score

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file", help="Input file")
    args = parser.parse_args()

    with open(args.input_file) as rf:
        data = json.load(rf)

    classes = [c[:9] for c in data["classes"]]

    print(f"Data size: {len(data['data'])}")

    for confidence_threshold in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

        labels = [
            sample["label"]
            for sample in data["data"]
            if sample["confidence"][sample["prediction"]] >= confidence_threshold
        ]
        preds = [
            sample["prediction"]
            for sample in data["data"]
            if sample["confidence"][sample["prediction"]] >= confidence_threshold
        ]
        if not labels or not preds:
            macro_f1 = "N/A"
            micro_f1 = "N/A"
        else:
            macro_f1 = f1_score(labels, preds, average="macro")
            micro_f1 = f1_score(labels, preds, average="micro")

        print(
            f"[Confidence {confidence_threshold}] Micro f1: {micro_f1}, Macro f1: {macro_f1}"
        )

    labels = [sample["label"] for sample in data["data"]]
    preds = [sample["prediction"] for sample in data["data"]]
    matrix = confusion_matrix(labels, preds)
    name_string = "".join(f"{name:10s}" for name in classes)
    print(f"{' '*10}{name_string}")
    for name, row in zip(classes, matrix):
        row_string = "".join(f"{str(v):10s}" for v in row)
        print(f"{name:10s}{row_string}")

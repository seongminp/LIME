import argparse
from collections import defaultdict
from pathlib import Path


def get_model_key(dataset, model, threshold, is_soft):
    model_key = f"{dataset}_{model}_{float(threshold)}"
    model_key = f"{model_key}_soft" if is_soft else model_key
    return model_key


def remove_outliers(array):
    # array.remove(max(array))
    # array.remove(min(array))
    return array


if __name__ == "__main__":

    # Example:
    #
    # python scripts/emnlp22/f1_stats.py -d results

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data-directory", help="Directory with training results"
    )
    args = parser.parse_args()

    data = defaultdict(dict)
    for data_file in Path(args.data_directory).glob("**/*.txt"):
        if "supervised" in str(data_file):
            continue
        with open(data_file) as rf:
            text = rf.readlines()[1]
        macro_text, micro_text = text.strip().split(",")
        macro = float(macro_text.strip())
        micro = float(micro_text.strip())

        properties = data_file.stem.split("_")
        dataset = properties[0]
        if "qa_what" in str(data_file) or "qa_article" in str(data_file):
            model = "_".join(properties[1:3])
        else:
            model = properties[1]
        is_soft = properties[-1] == "soft"
        threshold = float(properties[-2]) if is_soft else float(properties[-1])
        model_key = get_model_key(dataset, model, threshold, is_soft)

        data[model_key]["micro"] = micro
        data[model_key]["macro"] = macro

    models = ["entailment", "nsp", "rnsp", "qa", "qa_what", "xclass", "lotclass"]
    diffs = {
        model: {"macro_diff": [], "micro_diff": [], "avg_diff": []} for model in models
    }
    for dataset in ["agnews", "yahoo", "dbpedia", "tweet", "clickbait"]:
        print(f"[{dataset}]")
        for model in models:
            model_key_normal = get_model_key(dataset, model, 0, False)
            if not model_key_normal in data:
                continue

            micro_normal = data[model_key_normal]["micro"]
            macro_normal = data[model_key_normal]["macro"]
            avg_normal = (micro_normal + macro_normal) / 2

            model_key_soft = get_model_key(dataset, model, 0, True)
            micro_soft = data[model_key_soft]["micro"]
            macro_soft = data[model_key_soft]["macro"]
            avg_soft = (micro_soft + macro_soft) / 2

            micro_diff = micro_soft - micro_normal
            macro_diff = macro_soft - macro_normal
            avg_diff = avg_soft - avg_normal

            print(f"{model} - {micro_diff}, {macro_diff}, {avg_diff}")

            diffs[model]["micro_diff"].append(micro_diff)
            diffs[model]["macro_diff"].append(macro_diff)
            diffs[model]["avg_diff"].append(avg_diff)

    print("----")
    for model, diff_dict in diffs.items():
        micro, macro, avg = (
            remove_outliers(diff_dict["micro_diff"]),
            remove_outliers(diff_dict["macro_diff"]),
            remove_outliers(diff_dict["avg_diff"]),
        )
        print(
            model,
            sum(micro) / len(micro),
            sum(macro) / len(macro),
            sum(avg) / len(avg),
            len(micro),
        )

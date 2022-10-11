import argparse
import itertools
from collections import defaultdict
from pathlib import Path


def get_model_key(dataset, model, threshold, is_soft):
    model_key = f"{dataset}_{model}_{float(threshold)}"
    model_key = f"{model_key}_soft" if is_soft else model_key
    return model_key

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-directory", help="Data directory")
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
    # x_axis = np.arange(len(x))

    # models = ["entailment", "nsp", "rnsp", "qa", "xclass", "lotclass"]
    models = [
        "lotclass",
        "xclass",
        "entailment",
        # "nsp",
        "rnsp",
        "qa",
        # "qa_what",
        # "qa_article",
    ]
    model_types = ["", "soft"]
    # datasets = ["agnews", "yahoo", "dbpedia", "tweet", "clickbait"]
    datasets = ["agnews", "yahoo", "dbpedia", "clickbait"]
    dataset_text = "\t".join(f"{d:30s}" for d in datasets)
    title_text = f"{'MODEL':10s}\t{dataset_text}\n{'-'*120}"
    micro_diffs = {pair:0 for pair in itertools.product(datasets, models)}
    macro_diffs = {pair:0 for pair in itertools.product(datasets, models)}
    for model_type in model_types:
        row_texts = [title_text]
        print(model_type if model_type == "soft" else "hard")
        for model in models:
            table_row = [f"{model:10s}"]
            for dataset in datasets:
                is_soft = model_type == "soft"
                soft_model_key = get_model_key(dataset, model, 0, True)
                hard_model_key = get_model_key(dataset, model, 0, False)

                model_key = soft_model_key if model_type == "soft" else hard_model_key

                if model_key in data:
                    micro = data[model_key]["micro"]
                    macro = data[model_key]["macro"]

                    if model_type == "soft":
                        hard_micro = data[hard_model_key]["micro"]
                        hard_macro = data[hard_model_key]["macro"]
                        micro_diff = (micro - hard_micro) * 100
                        macro_diff = (macro - hard_macro) * 100
                        micro_sign = '+' if micro_diff >= 0 else '-'
                        macro_sign = '+' if macro_diff >= 0 else '-'
                        score_text = f"{micro*100:.2f} ({micro_sign}{abs(micro_diff):.2f}) / {macro*100:.2f} ({macro_sign}{abs(macro_diff):.2f})"
                        micro_diffs[(dataset, model)] += micro_diff
                        macro_diffs[(dataset, model)] += macro_diff
                    else:
                        score_text = f"{micro*100:.2f} / {macro*100:.2f}"
                else:
                    score_text = None
                table_row.append(score_text)
            #tab = "\t"
            tab = " & "
            row_text = f"{tab.join(table_row)}"
            row_texts.append(row_text)
        print("\n".join(row_texts) + "\n\n")

    
    for model in models:
        micro_sum = sum([micro_diffs[(dataset, model)] for dataset in datasets])
        macro_sum = sum([macro_diffs[(dataset, model)] for dataset in datasets])

        micro_avg = micro_sum / len(datasets)
        macro_avg = macro_sum / len(datasets) 
        print(f'{model}, ({micro_avg:.2f} / {macro_avg:.2f})')



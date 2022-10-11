import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from plot_constants import colors, markers, dataset_name
from plot_constants import model_name as model_names

CB91_Blue = "#2CBDFE"
CB91_Green = "#47DBCD"
CB91_Pink = "#F3A0F2"
CB91_Purple = "#9D2EC5"
CB91_Violet = "#661D98"
CB91_Amber = "#F5B14C"
color_list = [
    CB91_Purple,
    CB91_Green,
    CB91_Amber,
    CB91_Blue,
    CB91_Pink,
    CB91_Violet,
    "red",
    "green",
]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors.values())
plt.rcParams["font.family"] = "Times New Roman"

#markers = ["x", "+", "o", "8", "s", "X", "D", "p", "P", "d"]


params = {'mathtext.default': 'regular' } 
plt.rcParams.update(params)

x_axis = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def normalize(scores, reference):
    diff = reference - scores[5]
    new_scores = [score + diff for score in scores]
    return new_scores


def get_model_key(dataset, model, threshold, is_soft):
    model_key = f"{dataset}_{model}_{float(threshold)}"
    model_key = f"{model_key}_soft" if is_soft else model_key
    return model_key


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-directory", help="Data directory")
    parser.add_argument("-ds", "--dataset", help="Dataset name")
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
        "entailment",
        # "nsp",
        "rnsp",
        "qa",
        # "qa_what",
        # "qa_article",
        "xclass",
        "lotclass",
    ]
    model_types = ["", "soft"]
    # datasets = ["agnews", "yahoo", "dbpedia", "tweet", "clickbait"]
    # datasets = ["dbpedia"]
    #datasets = [args.dataset]
    datasets = ["agnews", "yahoo", "dbpedia", "clickbait"]
    to_plot = defaultdict(list)
    for dataset in datasets:
        for model in models:
            for model_type in model_types:
                vals = []
                is_soft = model_type == "soft"
                for threshold in x_axis:
                    model_key = get_model_key(dataset, model, threshold, is_soft)
                    if model_key in data:
                        micro = data[model_key]["micro"]
                        macro = data[model_key]["macro"]
                        # val = (micro + macro) / 2
                        val = macro
                    else:
                        val = None
                    vals.append(val)
                #model_name = f"{dataset}_{model}"
                model_name = model
                #if model_type:
                    #model_name += f"_{model_type}"
                to_plot[dataset].append((model_name, model, vals, model_type))

    fig, axs = plt.subplots(1, len(datasets), figsize=(20, 5))
    # ax.set_title(
    # f"Average F1 Score vs. Pseudo-Label Confidence Threshold",
    # fontweight="bold",
    # pad=30,
    # fontsize=20,
    # )
    # ax.set_ylim([0, 4])
    #ax.set_title(dataset_name[datasets[0]], fontsize=15)
    #for i, (name, model, vals) in enumerate(to_plot):
    for i, dataset in enumerate(datasets):
        ax = axs[i]
        macros = to_plot[dataset]
        for (model_name, model, vals, model_type) in macros:
            #marker = markers[models.index(model)]
            marker = markers[model_name]
            color = colors[model_name]
            line_style = ":" if model_type == "soft" else "-"
            ax.plot(
                x_axis,
                vals,
                line_style,
                label=model_names[model] if model_type != "soft" else None,
                marker=marker,
                c=color,
                markersize=5,
                # markerfacecolor="white"
            )
        ax.set_title(dataset_name[dataset], style="oblique", fontsize=15)
        ax.set_xlabel("Minimum label confidence", style="italic", fontsize=15, labelpad=10)
        ax.set_ylabel(f"$F_1$ score", style="italic", fontsize=15, labelpad=10)
        ax.set_xticks(x_axis)
        ax.set_xticklabels(
            ["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
        )
        ax.tick_params(axis="y", labelsize=15)
        ax.tick_params(axis="x", labelsize=15)

    ax.legend()

    # ax.bar_label(doc, padding=3)
    # ax.bar_label(dial, padding=3)

    # Cleanup.
    # ax.spines["top"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.spines["left"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(
        axis="both",
        which="both",
        bottom="off",
        top="off",
        labelbottom="on",
        left="off",
        right="off",
        labelleft="on",
        size=5,
    )

    name = "_".join(datasets)
    fig_name = f"{name.replace(' ', '_')}.svg"
    fig.tight_layout()
    fig.savefig(fig_name)
    print(f"Saved as {fig_name}")

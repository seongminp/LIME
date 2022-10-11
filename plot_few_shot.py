import argparse
import json
import random

import matplotlib
import matplotlib.pyplot as plt
from plot_constants import colors, markers, model_name
from sklearn.metrics import confusion_matrix, f1_score

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

# markers = ["x", "+", "o", "8", "s", "X", "D", "p", "P", "d"]


def normalize(array):
    norm = [float(i) / sum(array) for i in array]
    return norm


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    maxes = {
        "entailment": 0.6007,
        "nsp": 0.6265,
        "rnsp": 0.6190,
        "qa": 0.8067,
        "qa_what": 0.40052,
        "xclass": 0.5780,
    }

    # x_axis = [10, 50, 100, 200, 500, 1000, 2000, 5000, 7000]
    x_axis = [10, 50, 100, 200, 500, 1000, 3000]
    xclass = [
        0.1,
        0.1955,
        0.161014,
        0.2233517,
        0.259477,
        0.4003,
        0.5146,
        0.5146,
        0.633,
    ][: len(x_axis)]

    data = {"xclass": xclass}
    # models = ["entailment", "rnsp", "qa", "qa_what", "qa_article"]
    models = ["entailment", "rnsp", "qa"]
    for model in models:

        data_file = f"./data/agnews/preds_{model}.json"
        with open(data_file) as rf:
            model_data = json.load(rf)

        model_data = model_data["data"]
        model_f1 = [0] * len(x_axis)
        seeds = list(range(2, 8))
        for seed in seeds:
            random.seed(seed)
            random.shuffle(model_data)
            for i, data_size in enumerate(x_axis):
                subdata = model_data[:data_size]
                labels = [sample["label"] for sample in subdata]
                preds = [sample["prediction"] for sample in subdata]
                macro_f1 = f1_score(labels, preds, average="macro")
                model_f1[i] += macro_f1
        model_f1 = [f1 / len(seeds) for f1 in model_f1]
        data[model] = model_f1

    for model, f1s in data.items():
        print(model, f1s)

    fig, ax = plt.subplots(figsize=(4, 3))
    # ax.set_title(
    # f"Pseudo-label F1 vs. Test Set Size",
    # fontweight="bold",
    # pad=30,
    # fontsize=20,
    # )
    # ax.set_ylim([0, 4])
    ax.set_xlabel("Dataset size", style="italic", fontsize=15, labelpad=10)
    ax.set_ylabel(f"Weak label macro-$F_1$", style="italic", fontsize=15, labelpad=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_xscale("log")

    models.append("xclass")
    for model, vals in data.items():
        marker = markers[model]
        # marker = "o"
        color = "red" if model == "xclass" else CB91_Blue
        # color = CB91_Blue if model == 'xclass' else 'grey'
        # color = color_list[models.index(model)]
        if model == "entailment":
            line_style = "dashed"
        elif model == "rnsp":
            line_style = "dotted"
        elif model == "qa":
            line_style = "dashdot"
        elif model == "xclass":
            line_style = "solid"
        line_style = "solid"

        # vals = normalize(vals)
        max = maxes[model]
        # vals = [v/max for v in vals + [max]]
        vals = vals + [max]
        x = x_axis + [7600]
        # x = x_axis
        lw = 2 if model == "xclass" else 1
        markersize = 4 if model == "xclass" else 3
        ax.plot(
            x,
            vals,
            linestyle=line_style,
            label=model_name[model],
            marker=marker,
            markersize=markersize,
            markerfacecolor="white",
            lw=lw,
            c=color,
        )

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

    params = {'mathtext.default': 'regular' } 
    plt.rcParams.update(params)
    ax.set_xticks(x_axis + [7600])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticklabels(x_axis + ["Full"])

    ax.get_xaxis().set_tick_params(which="minor", size=0)
    ax.get_xaxis().set_tick_params(which="minor", width=0)

    name = "pseudo_f1s"
    fig_name = f"{name.replace(' ', '_')}.svg"
    fig.tight_layout()
    fig.savefig(fig_name)
    print(f"Saved as {fig_name}")

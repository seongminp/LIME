import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_constants import colors, markers, model_name

CB91_Blue = "#2CBDFE"
CB91_Green = "#47DBCD"
CB91_Pink = "#F3A0F2"
CB91_Purple = "#9D2EC5"
CB91_Violet = "#661D98"
CB91_Amber = "#F5B14C"
# color_list = [CB91_Purple, CB91_Green, CB91_Amber, CB91_Blue, CB91_Pink, CB91_Violet]
color_list = [CB91_Blue, CB91_Pink, CB91_Amber, CB91_Green, CB91_Pink, CB91_Violet]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors.values())
plt.rcParams["font.family"] = "Times New Roman"

# markers = ["x", "+", "o", "8", "s", "X", "D", "p", "P", "d"]


x_axis = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def get_ax_index(name):
    if name == "agnews":
        return 0
    if name == "yahoo":
        return 1
    if name == "dbpedia":
        return 1
    if name == "tweet":
        return 1
    if name == "clickbait":
        return 1


def get_marker_index(name):
    if name == "entailment":
        return 0
    if name == "nsp":
        return 1
    if name == "rnsp":
        return 2
    if name == "qa":
        return 4
    if name == "xclass":
        return 5
    if name == "lotclass":
        return 6


def get_color_index(name):
    if name == "entailment":
        return 0
    if name == "nsp":
        return 0
    if name == "rnsp":
        return 0
    if name == "qa":
        return 0
    if name == "xclass":
        return -1
    if name == "lotclass":
        return -1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument("-d", "--data-directory", help="Data directory")
    args = parser.parse_args()

    data = []
    counts = {}
    # models = ["entailment", "nsp", "rnsp", "qa", "xclass", "lotclass"]
    #models = ["entailment", "rnsp", "qa", "xclass", "lotclass"]
    models = ["lotclass", "xclass", "entailment", "rnsp", "qa"]
    datasets = ["agnews"]
    for dataset in datasets:
        for model in models:
            data_file = f"data/{dataset}/preds_{model}.json"
            with open(data_file) as rf:
                model_data = json.load(rf)
            count_list = []
            for confidence in x_axis:
                count = len(
                    [
                        s
                        for s in model_data["data"]
                        if s["confidence"][s["prediction"]] >= confidence
                    ]
                )
                count_list.append(count)
            count_list = np.array(count_list)
            counts[f"{dataset}_{model}"] = count_list

    # fig, ax = plt.subplots(1, 5, figsize=(20, 5))
    # fig, ax = plt.subplots(1, 2, figsize=(8, 5))
    fig, ax = plt.subplots(figsize=(4, 3))
    # fig.suptitle(
    # f"Pseudo-Label Example Counts",
    # fontweight="bold",
    # pad=30,
    # fontsize=20,
    # )
    # ax.set_ylim([0, 4])
    # ax.set_xlabel("Pseudo label confidence", style="italic", fontsize=20, labelpad=10)
    # ax.set_ylabel(f"Count", style="italic", fontsize=20, labelpad=10)
    # ax.set_xticks(x_axis)
    # ax.set_xticklabels(
    # ["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
    # )
    # ax.tick_params(axis="y", labelsize=15)
    # ax.tick_params(axis="x", labelsize=15)

    params = {'mathtext.default': 'regular' } 
    plt.rcParams.update(params)
    for dataset in datasets:
        print(f"Plotting {dataset}...")
        col = get_ax_index(dataset)
        ax.set_xlabel(f"Confidence", style="italic", fontsize=15, labelpad=10)
        ax.set_ylabel("Count", style="italic", fontsize=15, labelpad=10)
        # ax[col].set_title(dataset, style="italic", fontsize=15, fontweight="bold")
        ax.set_xticks(x_axis)
        ax.set_xticklabels(
            ["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
        )
        for model in models:
            if model == "xclass" or model == "lotclass":
                color = "red"
                width = 2
                markersize = 6
            else:
                color = CB91_Blue
                width = 1
                markersize = 4
            marker = markers[model]
            print(count_list)
            count_list = counts[f"{dataset}_{model}"]
            ax.plot(
                x_axis,
                count_list,
                color=color,
                marker=marker,
                label=model_name[model],
                lw=width,
                markersize=markersize,
            )
            print(f"\tFinished {model}.")
            ax.legend()

    plt.legend()
    # Cleanup.
    # ax.spines["top"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.spines["left"].set_visible(False)
    # ax.get_xaxis().tick_bottom()
    # ax.get_yaxis().tick_left()
    # ax.tick_params(
    # axis="both",
    # which="both",
    # bottom="off",
    # top="off",
    # labelbottom="on",
    # left="off",
    # right="off",
    # labelleft="on",
    # size=5,
    # )

    name = "pseudo_labels_count"
    fig_name = f"{name.replace(' ', '_')}.svg"
    fig.tight_layout()
    fig.savefig(fig_name)
    print(f"Saved as {fig_name}")

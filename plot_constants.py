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

color_list = [
    "#e6194B",  # Red
    "#3cb44b",  # Green
    "#f58231",  # Orange
    "#4363d8",  # Dark blue
    "#911eb4",  # Purple
    "#f032e6",  # Pink
    "#42d4f4",
    "#bfef45",
    "#fabed4",
    "#469990",
    "#dcbeff",
    "#9A6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#a9a9a9",
    "#ffffff",
    "#000000",
    "#ffe119",
]

# colors = {
# "entailment": cb91_purple,
# "nsp":  cb91_green,
# "rnsp": cb91_violet,
# "qa": cb91_blue,
# "xclass": cb91_amber,
# "lotclass": cb91_pink
# }
colors = {
    "entailment": color_list[0],
    "rnsp": color_list[1],
    "qa": color_list[2],
    "xclass": color_list[3],
    "lotclass": color_list[4],
    "nsp": color_list[5],
    "qa_what": color_list[6],
    "qa_article": color_list[7],
}

markers = {
    "entailment": "o",
    "nsp": "^",
    "rnsp": "^",
    "qa": "D",
    "xclass": "x",
    "lotclass": "+",
    "qa_what": "s",
}

model_name = {
    "entailment": "$PLAT_{ENT}$",
    "nsp": "PLAT_{nsp}",
    "rnsp": "$PLAT_{NSP}$",
    "qa": "$PLAT_{QA}$",
    "xclass": "X-Class",
    "lotclass": "LOTClass",
    "qa_what": "s",
}

dataset_name = {
    "agnews": "AGNews",
    "yahoo": "Yahoo",
    "dbpedia": "DBpedia",
    "clickbait": "Clickbait",
}

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

import matplotlib.pyplot as plt
import subprocess
import os
import pandas as pd

METHODS = ["bow", "ftc", "sbert"]
MODELS = ['bing', 'google', 'opus']
MUTATION_TYPE = ["gender", "negative", "plural", "tense"]

def draw_precision_plot(precisions_list, fig_path):
    fig, ax = plt.subplots()
    x = range(50)
    ax.plot(x, precisions_list[0], label="Bing")
    ax.plot(x, precisions_list[1], label="Google")
    ax.plot(x, precisions_list[2], label="OPUS")
    ax.plot(x, precisions_list[3], label="Average")
    ax.set_xlabel("Alignment window size", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.legend()
    plt.savefig(fig_path)

def draw_count_plot(count_list, fig_path):
    fig, ax = plt.subplots()
    x = range(50)
    ax.plot(x, count_list[0], label="Bing")
    ax.plot(x, count_list[1], label="Google")
    ax.plot(x, count_list[2], label="OPUS")
    ax.plot(x, count_list[3], label="Average")
    ax.set_xlabel("Alignment window size", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend()
    plt.savefig(fig_path)

if __name__ == '__main__':
    models = ["bing", "google", "opus"]
    precisions_list = list()
    totals = list()
    sense_inventory_path = './asset/sense_inventory/sense_dict.json'

    draw_df = {
        "method": [],
        "model": [],
        "aws": [],
        "positive": [],
        "true_positive": [],
    }

    for method in METHODS:
        print(method)
        for model in models:
            print(model)
            for aws in range(50):
                df_list = []
                for mut in MUTATION_TYPE:
                    input_path = f"./rq/rq4/golden-dataset-{model}-{mut}_{method}_{aws}.csv"
                    
                    df = pd.read_csv(input_path)
                    df_list.append(df)
                
                all_df = pd.concat(df_list)
                true_positive = len(all_df[(all_df['ori.sc_id'] >= 0) & (all_df['mut.sc_id'] >= 0)
                                    & (all_df['ori.sc_id'] != all_df['mut.sc_id']) & (all_df['unfair'] > 0)])
                positive = len(all_df[(all_df['ori.sc_id'] >= 0) & (all_df['mut.sc_id'] >= 0) & (all_df['ori.sc_id'] != all_df['mut.sc_id'])])
                
                draw_df["method"].append(method)
                draw_df["model"].append(model)
                draw_df["aws"].append(aws)
                draw_df["positive"].append(positive)
                draw_df["true_positive"].append(true_positive)
    
    draw_df = pd.DataFrame(draw_df)
    draw_df.to_csv("rq/rq4/rq4.csv", index=False)

    for method in METHODS:
        sub_df = draw_df[draw_df["method"] == method].copy()
        sub_df["precision"] = sub_df["true_positive"] / sub_df["positive"]

        precisions_list = []
        for model in models:
            precisions_list.append(sub_df[sub_df["model"] == model].precision.values)

        precisions_list.append(sub_df.sort_values("aws").groupby("aws").agg("mean").precision)
        fig_path = f"rq/rq4/rq4_{method}_precision.pdf"
        draw_precision_plot(precisions_list, fig_path)

        count_list = []
        for model in models:
            count_list.append(sub_df[sub_df["model"] == model].positive.values)

        count_list.append(sub_df.sort_values("aws").groupby("aws").agg("mean").positive)
        fig_path = f"rq/rq4/rq4_{method}_positive.pdf"
        draw_count_plot(count_list, fig_path)

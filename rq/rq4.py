import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

import os
import pandas as pd
import matplotlib.pyplot as plt
from fairwsd4mt import venn
import tqdm

def main():
    mutation_types = ["gender", "negative", "plural", "tense"]

    fig=plt.figure(figsize=(21,7))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax_list = [ax1, ax2, ax3]

    model_names = ["(a) Google", "(b) Bing", "(c) OPUS"]

    for model_i, model in enumerate(["google", "bing", "opus"]):
        labels = {}
        for m in mutation_types:
            labels[m] = []

        for mutation_type in mutation_types:
            print(model, mutation_type)
            path = f"result/{model}/test-set-{mutation_type}_{model}_merged_bow_49.csv"
            df = pd.read_csv(path)

            true_positive_df = df[(df['ori.sc_id'] >= 0) & (df['mut.sc_id'] >= 0) & (df['ori.sc_id'] != df['mut.sc_id'])]

            for i, row in tqdm.tqdm(true_positive_df.iterrows(), total=len(true_positive_df)):
                labels[mutation_type].append(str(row["ori.src"]))
        labels = venn.get_labels([labels["gender"], labels["negative"], labels["plural"], labels["tense"]], fill=['number'])
        ax = venn.venn4_ax(ax_list[model_i], labels, names=['Gender', 'Positive/Negative', 'Singular/Plural', 'Tense'], legend=(model_i==2), fontsize=11)
        ax.set_title(model_names[model_i], y=-0.01, fontdict={'fontsize':16})

    fig.savefig(f"venn_out.pdf")

if __name__ == "__main__":
    main()
import os
import pandas as pd

def main():
    out = {
        "mutation_type": [],
        "model": [],
        "mutkind": [],
        "correct": [],
        "number": [],
        "ratio": [],
    }
    for mutation_type in ["gender", "negative", "plural", "tense"]:
        for model in ["google", "bing", "opus"]:
            path = f"result/{model}/test-set-{mutation_type}_{model}_merged_bow_49.csv"
            df = pd.read_csv(path)

            mutkinds = df["mutkind"].unique()

            for mutkind in mutkinds:
                sub_df = df[df["mutkind"] == mutkind]

                true_positive = len(sub_df[(sub_df['ori.sc_id'] >= 0) & (sub_df['mut.sc_id'] >= 0) & (sub_df['ori.sc_id'] != sub_df['mut.sc_id'])])

                out["mutation_type"].append(mutation_type)
                out["model"].append(model)
                out["mutkind"].append(mutkind)
                out["correct"].append(true_positive)
                out["number"].append(len(sub_df))
                out["ratio"].append(true_positive/len(sub_df))
    
    out_df = pd.DataFrame(out)
    out_df.to_csv("rq3.csv", index=False)

if __name__ == "__main__":
    main()
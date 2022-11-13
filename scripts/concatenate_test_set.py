import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

import pandas as pd
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mutation_type', '-m', type=str, required=True,
                        help='mutation type')
    args = parser.parse_args()

    # concatenate corpus to build a test set
    um = pd.read_csv(f'./asset/corpus/um_{args.mutation_type}_mutated.csv')
    os18 = pd.read_csv(f'./asset/corpus/os18_{args.mutation_type}_mutated.csv')
    ncv15 = pd.read_csv(f'./asset/corpus/nc-v15_{args.mutation_type}_mutated.csv')
    cwmt = pd.read_csv(f'./asset/corpus/cwmt_{args.mutation_type}_mutated.csv')
    UN = pd.read_csv(f'./asset/corpus/UNv1_{args.mutation_type}_mutated.csv')

    total = pd.concat([um, os18, ncv15, cwmt, UN], ignore_index=True)
    total = total[total["tag"] == "MUTATED"]
    total.to_csv(f'./test-set-{args.mutation_type}.csv', index=False)





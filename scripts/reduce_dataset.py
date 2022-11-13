
import pandas as pd
import argparse

def sample(sentence_info_path: str, output_path: str, sample_size: int):
    df = pd.read_csv(sentence_info_path)
    l = len(df)
    s = min(sample_size, l)
    reduced_df = df.sample(n=s, random_state=0)
    reduced_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentence_info_path', '-s', type=str, required=True,
                        help='path to csv file containing mutated sentence and related info')
    parser.add_argument('--output_path', '-o', type=str, required=True,
                        help='path to csv file containing mutated sentence and related info')
    parser.add_argument('--sample_size', '-n', type=int, required=False, default=100000,
                        help='number of sentences to be sampled')
    args = parser.parse_args()

    sample(args.sentence_info_path, args.output_path, args.sample_size)
    
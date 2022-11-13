import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

import os.path
import pandas as pd

def random_sample(mutation_result_path: str, mutation_result_samples_path: str):
    
    if not os.path.exists(mutation_result_samples_path):
        # Randomly sample 400 sentence pairs from mutation result
        mutation_result = pd.read_csv(mutation_result_path)
        mutation_result_samples = mutation_result.sample(400, random_state=46)
        mutation_result_samples.to_csv(mutation_result_samples_path, index=False)
        print(f"Mutation result samples are saved to {mutation_result_samples_path}. Please manual check each sample and add the isOK flag.")
        return
    else:
        mutation_result_samples = pd.read_csv(mutation_result_samples_path)

    # Has isOK column
    if 'isOK' not in mutation_result_samples.columns:
        mutation_result_samples['isOK'] = 0
        mutation_result_samples.to_csv(mutation_result_samples_path, index=False)
        print(f"Mutation result samples are saved to {mutation_result_samples_path}. Please manual check each sample and add the isOK flag.")
        return

    realistic_df = mutation_result_samples[mutation_result_samples['isOK'] == 1]
    grammatical_error_df = mutation_result_samples[mutation_result_samples['isOK'] == -4]
    counterintuitive_df = mutation_result_samples[mutation_result_samples['isOK'] == -2]

    total_count = len(mutation_result_samples)
    realistic_count = len(realistic_df)
    grammatical_error_count = len(grammatical_error_df)
    counterintuitive_count = len(counterintuitive_df)

    print('Type\tCount\tRatio')
    print('Realistic', realistic_count, realistic_count / total_count)
    print('Grammatical error', grammatical_error_count, grammatical_error_count / total_count)
    print('Counterintuitive', counterintuitive_count, counterintuitive_count / total_count)


# RQ1: effectiveness of mutation
def rq1():
    random_sample("./test-set-gender.csv", './rq/rq1/mutation_samples_gender.csv')
    random_sample("./test-set-negative.csv", './rq/rq1/mutation_samples_negative.csv')
    random_sample("./test-set-plural.csv", './rq/rq1/mutation_samples_plural.csv')
    random_sample("./test-set-tense.csv", './rq/rq1/mutation_samples_tense.csv')

if __name__ == "__main__":
    rq1()


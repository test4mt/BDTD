import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

import time
from typing import Iterable, List
import pandas as pd
import argparse
import spacy
import re
from tqdm import tqdm

from fairwsd4mt.mutation.mutate import mutate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentence_info_path', '-s', type=str, required=True,
                        help='path to the source side of the parallel corpus to be mutated')
    parser.add_argument('--mutation_type', '-m', type=str, required=True, help='mutation type')
    args = parser.parse_args()

    mutate(args.sentence_info_path, args.mutation_type)
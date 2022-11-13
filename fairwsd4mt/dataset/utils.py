

import json
from typing import Any, Dict, Iterable, List, Tuple
import joblib
import pandas as pd
import tqdm

from fairwsd4mt.utils import chunked

def read_line_dataset(in_path: str) -> List[str]:
    with open(in_path, 'r', encoding='utf8') as fp:
        lines = [line.strip() for line in fp]
    return lines

def read_double_line_dataset(in_path: str) -> Tuple[List[str], List[str]]:
    first = []
    second = []
    with open(in_path, 'r', encoding='utf8') as fp:
        
        for i, line in enumerate(fp):
            if i % 2 == 0:
                first.append(line.strip())
            else:
                second.append(line.strip())
    return first, second

def read_tsv_dataset(in_path: str) -> pd.DataFrame:
    df = pd.read_csv(in_path, sep='\t', dtype=str)
    return df

def read_jsonl(path: str, progress_bar: bool=False):
    dataset = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

        if progress_bar:
            lines = tqdm.tqdm(lines)
        for line in lines:
            dataset.append(json.loads(line))
    return dataset

def write_jsonl(obj_list: Iterable[Any], output_path: str, progress_bar: bool=False):
    with open(output_path, 'w', encoding='utf-8') as f:
        if progress_bar:
            obj_list = tqdm.tqdm(obj_list)
        for each_obj in obj_list:
            f.write(json.dumps(each_obj, ensure_ascii=False) + '\n')


import gzip
def read_jsonl_gz(path: str, progress_bar: bool=False) -> List[Any]:
    dataset = []
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        lines = f.readlines()

        if progress_bar:
            lines = tqdm.tqdm(lines)
        for line in lines:
            dataset.append(json.loads(line))
    return dataset

def write_jsonl_gz(obj_list: Iterable[Any], output_path: str, progress_bar: bool=False):
    with gzip.open(output_path, 'wt', encoding='utf-8') as f:
        if progress_bar:
            obj_list = tqdm.tqdm(obj_list)
        for each_obj in obj_list:
            f.write(json.dumps(each_obj, ensure_ascii=False) + '\n')

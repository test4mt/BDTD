import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from typing import List
import tqdm
import pandas as pd

from fairwsd4mt.dataset.utils import read_line_dataset, read_tsv_dataset, read_double_line_dataset

def write_dataset(dataset: List[str], path: str):
    with open(path, "w", encoding='utf-8') as f:
        for line in dataset:
            f.write(line + "\n")

def extract_cwmt(in_path: str, src_out: str, trg_out: str):
    file_list = [
        ("cwmt/parallel/casia2015/casia2015_en.txt","cwmt/parallel/casia2015/casia2015_ch.txt"),
        ("cwmt/parallel/casict2011/casict-A_en.txt","cwmt/parallel/casict2011/casict-A_ch.txt"),
        ("cwmt/parallel/casict2011/casict-B_en.txt","cwmt/parallel/casict2011/casict-B_ch.txt"),
        ("cwmt/parallel/casict2015/casict2015_en.txt","cwmt/parallel/casict2015/casict2015_ch.txt"),
        ("cwmt/parallel/datum2015/datum_en.txt","cwmt/parallel/datum2015/datum_ch.txt"),
        ("cwmt/parallel/datum2017/Book10_en.txt","cwmt/parallel/datum2017/Book10_cn.txt"),
        ("cwmt/parallel/datum2017/Book11_en.txt","cwmt/parallel/datum2017/Book11_cn.txt"),
        ("cwmt/parallel/datum2017/Book12_en.txt","cwmt/parallel/datum2017/Book12_cn.txt"),
        ("cwmt/parallel/datum2017/Book13_en.txt","cwmt/parallel/datum2017/Book13_cn.txt"),
        ("cwmt/parallel/datum2017/Book14_en.txt","cwmt/parallel/datum2017/Book14_cn.txt"),
        ("cwmt/parallel/datum2017/Book15_en.txt","cwmt/parallel/datum2017/Book15_cn.txt"),
        ("cwmt/parallel/datum2017/Book16_en.txt","cwmt/parallel/datum2017/Book16_cn.txt"),
        ("cwmt/parallel/datum2017/Book17_en.txt","cwmt/parallel/datum2017/Book17_cn.txt"),
        ("cwmt/parallel/neu2017/NEU_en.txt","cwmt/parallel/neu2017/NEU_cn.txt"),
    ]

    source_dataset = []
    target_dataset = []
    for source_path, target_path in file_list:
        print(source_path, target_path)
        source_dataset += read_line_dataset(os.path.join(in_path, source_path))
        target_dataset += read_line_dataset(os.path.join(in_path, target_path))
    
    write_dataset(source_dataset, src_out)
    write_dataset(target_dataset, trg_out)


def extract_opensubtitle(in_path: str, src_out: str, trg_out: str):
    file_list = [
        ("OpenSubtitles.en-zh_cn.en", "OpenSubtitles.en-zh_cn.zh_cn")
    ]

    source_dataset = []
    target_dataset = []
    for source_path, target_path in file_list:
        print(source_path, target_path)
        source_dataset += read_line_dataset(os.path.join(in_path, source_path))
        target_dataset += read_line_dataset(os.path.join(in_path, target_path))
    
    write_dataset(source_dataset, src_out)
    write_dataset(target_dataset, trg_out)

def extract_news_commentary_v15(in_path: str, src_out: str, trg_out: str):
    file_list = [
        "news-commentary-v15.en-zh.tsv",
    ]

    source_dataset = []
    target_dataset = []
    for file_path in file_list:
        print(file_path)
        out = read_tsv_dataset(os.path.join(in_path, file_path))
        source_dataset += [str(sentence) for sentence in out.iloc[:, 0].tolist()]
        target_dataset += [str(sentence) for sentence in out.iloc[:, 1].tolist()]
    
    write_dataset(source_dataset, src_out)
    write_dataset(target_dataset, trg_out)

def extract_UNv1(in_path: str, src_out: str, trg_out: str):
    file_list = [
        ("en-zh/UNv1.0.en-zh.en", "en-zh/UNv1.0.en-zh.zh")
    ]

    source_dataset = []
    target_dataset = []
    for source_path, target_path in file_list:
        print(source_path, target_path)
        source_dataset += read_line_dataset(os.path.join(in_path, source_path))
        target_dataset += read_line_dataset(os.path.join(in_path, target_path))
    
    write_dataset(source_dataset, src_out)
    write_dataset(target_dataset, trg_out)

def extract_umcorpus(in_path: str, src_out: str, trg_out: str):
    file_list = [
        "UM-Corpus/data/Bilingual/Education/Bi-Education.txt",
        "UM-Corpus/data/Bilingual/Laws/Bi-Laws.txt",
        "UM-Corpus/data/Bilingual/Microblog/Bi-Microblog.txt",
        "UM-Corpus/data/Bilingual/News/Bi-News.txt",
        "UM-Corpus/data/Bilingual/Science/Bi-Science.txt",
        "UM-Corpus/data/Bilingual/Spoken/Bi-Spoken.txt",
        "UM-Corpus/data/Bilingual/Subtitles/Bi-Subtitles.txt",
        "UM-Corpus/data/Bilingual/Thesis/Bi-Thesis.txt",
    ]

    source_dataset = []
    target_dataset = []
    for file_path in file_list:
        print(file_path)
        src, trg = read_double_line_dataset(os.path.join(in_path, file_path))
        source_dataset += src
        target_dataset += trg
    
    write_dataset(source_dataset, src_out)
    write_dataset(target_dataset, trg_out)

def main():
    extract_cwmt("./asset/dataset/China Workshop on Machine Translation", "./asset/corpus/cwmt.src.txt", "./asset/corpus/cwmt.trg.txt")
    extract_opensubtitle("./asset/dataset/OpenSubtitles-v2018-en-zh_cn.txt", "./asset/corpus/os18.src.txt", "./asset/corpus/os18.trg.txt")
    extract_news_commentary_v15("./asset/dataset/news-commentary-v15.en-zh.tsv", "./asset/corpus/nc-v15.src.txt", "./asset/corpus/nc-v15.trg.txt")
    extract_UNv1("./asset/dataset/UNv1.0.en-zh", "./asset/corpus/UNv1.src.txt", "./asset/corpus/UNv1.trg.txt")
    extract_umcorpus("./asset/dataset/umcorpus-v1", "./asset/corpus/um.src.txt", "./asset/corpus/um.trg.txt")

if __name__ == "__main__":
    main()
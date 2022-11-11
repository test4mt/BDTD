# BDTD

This is the source code for the paper "Back Deduction based Testing for Word Sense Disambiguation Ability of Machine Translation Systems".

## Package Requirement

To run this code, some packages are needed as following:

```python
gensim==4.2.0
matplotlib==3.5.2
matplotlib-venn==0.11.7
numpy==1.23.1
opencc-python-reimplemented==0.1.6
pandas==1.4.3
requests==2.27.1
sacrebleu==2.1.0
scikit-learn==1.1.1
scipy==1.8.1
sentence-transformers==2.2.2
spacy==3.4.0
tqdm==4.63.0
transformers==4.20.1
wefe==0.3.2
spacy==3.4.0
google-cloud-translate==2.0.1
tenacity==8.0.1
pyside6==6.3.0
hanlp==2.1.0b37
selenium==4.4.0
sacremoses==0.0.53
```

All code in this repository are test under the environment of `Python 3.8.13`.

## Prepare dataset
Due to the licenses of the original NLP datasets, we do not provide download links here, please submit applications and download from the following projects.

The list of datasets:
* UM-Corpus
* CWMT
* OpenSubtitles2018
* News-Commentary v15
* United Nations Parallel Corpus v1.0

Unzip datasets to the path `asset/dataset`

## Extract and Clean Datasets
Most experiment-related scripts are saved under the path `./scripts`.

Here we define some variables:
```bash
models=("google" "bing" "baidu" "opus")
muttype=("gender" "plural" "tense" "negative")
```

First, we extract the dataset of various formats to the format of `csv`.
```bash
python scripts/extract_dataset.py
```

Then, we clean the dataset.
```bash
python scripts/clean.py --src_path asset/corpus/cwmt.src.txt --ref_path asset/corpus/cwmt.trg.txt --output_directory ./asset/corpus --sense_inventory asset/sense_inventory/sense_dict.json
python scripts/clean.py --src_path asset/corpus/nc-v15.src.txt --ref_path asset/corpus/nc-v15.trg.txt --output_directory ./asset/corpus --sense_inventory asset/sense_inventory/sense_dict.json
python scripts/clean.py --src_path asset/corpus/os18.src.txt --ref_path asset/corpus/os18.trg.txt --output_directory ./asset/corpus --sense_inventory asset/sense_inventory/sense_dict.json
python scripts/clean.py --src_path asset/corpus/um.src.txt --ref_path asset/corpus/um.trg.txt --output_directory ./asset/corpus --sense_inventory asset/sense_inventory/sense_dict.json
python scripts/clean.py --src_path asset/corpus/UNv1.src.txt --ref_path asset/corpus/UNv1.trg.txt --output_directory ./asset/corpus --sense_inventory asset/sense_inventory/sense_dict.json
```

Mutate each dataset sequentially using the four mutation operators.
```bash
for i in "${muttype[@]}"
do
    echo "Mutate cwmt $i"
    python scripts/mutate.py --sentence_info_path asset/corpus/cwmt.csv --mutation_type $i
    echo "Mutate nc-v15 $i"
    python scripts/mutate.py --sentence_info_path asset/corpus/nc-v15.csv --mutation_type $i
    echo "Mutate os18 $i"
    python scripts/mutate.py --sentence_info_path asset/corpus/os18.csv --mutation_type $i
    echo "Mutate um $i"
    python scripts/mutate.py --sentence_info_path asset/corpus/um.csv --mutation_type $i
    echo "Mutate UNv1 $i"
    python scripts/mutate.py --sentence_info_path asset/corpus/UNv1.csv --mutation_type $i
done
```

Concatenate all datasets and limit the size.
```bash
# concatenate
echo "Concatenating datasets..."
for i in "${muttype[@]}"
do
    echo "Concatenate $i"
    python scripts/concatenate_test_set.py --mutation_type $i
done

# sample small test set
echo "Reduce datasets..."
for t in "${muttype[@]}"
do
    mv test-set-${t}.csv test-set-${t}-full.csv
    python scripts/reduce_dataset.py --sentence_info_path test-set-${t}-full.csv --output_path test-set-${t}.csv --sample_size 300000
done
```

Translate original sentences and mutated sentences.
```bash
# translate
echo "Translating datasets..."
for t in "${muttype[@]}"
do
    for m in "${models[@]}"
    do
        echo "Translate $t mutated dataset by $m"
        python scripts/translate.py --sentence_info_path test-set-$t.csv --model $m --mutation_type $t
    done
done
```

Align original sentences and mutated sentences.
```bash
# align
echo "Aligning datasets..."
for t in "${muttype[@]}"
do
    for m in "${models[@]}"
    do
        echo "Align $t mutated dataset by $m"
        python scripts/align.py --sentence_info_path test-set-$t.csv --input_path result/$m/test-set-${t}_${m}.csv --src_side src --tgt_side tgt --alignment_tool_path ../fast_align/build/
    done
done
```

Assign sense to the homograph.
```bash
# assign
echo "Assigning datasets..."
for t in "${muttype[@]}"
do
    for m in "${models[@]}"
    do
        python scripts/assign.py --input_path result/$m/test-set-${t}_${m}_merged.csv --method bow --sense_inventory asset/sense_inventory/sense_dict.json --alignment_window_size 0
        python scripts/assign.py --input_path result/$m/test-set-${t}_${m}_merged.csv --method ftc --sense_inventory asset/sense_inventory/sense_dict.json --alignment_window_size 0
        python scripts/assign.py --input_path result/$m/test-set-${t}_${m}_merged.csv --method sbert --sense_inventory asset/sense_inventory/sense_dict.json --alignment_window_size 49
    done
done
```


## Research Questions
All scripts about RQs are saved under the path `./rq`.

## References

For more details about data processing, please refer to the `code comments` and our paper.



from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianTokenizer
OPUS_MODEL = None
OPUS_TOKENIZER = None

# from sacrebleu.tokenizers import tokenizer_zh
# char_zh_tokenize = tokenizer_zh.TokenizerZh()

GPU_DEVICE = "cuda"

def translate_text_with_opus(text: str) -> str:
    global OPUS_MODEL, OPUS_TOKENIZER
    if OPUS_MODEL is None:
        OPUS_MODEL = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
        print("CUDA", torch.cuda.is_available())
        if torch.cuda.is_available():
            OPUS_MODEL = OPUS_MODEL.to(GPU_DEVICE)
    model = OPUS_MODEL
    
    if OPUS_TOKENIZER is None:
        OPUS_TOKENIZER = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
    tokenizer = OPUS_TOKENIZER

    tokens = tokenizer(text, return_tensors="pt", padding=True)
    if torch.cuda.is_available():
        tokens = {k: v.to(GPU_DEVICE) for k, v in tokens.items()}
    translated = model.generate(**tokens)
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated][0]

def translate_text_with_opus_batch(text: List[str]) -> List[str]:
    global OPUS_MODEL, OPUS_TOKENIZER
    if OPUS_MODEL is None:
        OPUS_MODEL = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
        print("CUDA", torch.cuda.is_available())
        if torch.cuda.is_available():
            OPUS_MODEL = OPUS_MODEL.to(GPU_DEVICE)
    model = OPUS_MODEL
    
    if OPUS_TOKENIZER is None:
        OPUS_TOKENIZER = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
    tokenizer = OPUS_TOKENIZER

    tokens = tokenizer(text, return_tensors="pt", padding=True)
    if torch.cuda.is_available():
        tokens = {k: v.to(GPU_DEVICE) for k, v in tokens.items()}
    translated = model.generate(**tokens)
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

import os
from typing import List
from joblib import Parallel, delayed

import os
import json

"""Translates text into the target language.

Make sure your project is allowlisted.

Target must be an ISO 639-1 language code.
See https://g.co/cloud/translate/v2/translate-reference#supported_languages

Copied from https://cloud.google.com/translate/docs/samples/translate-text-with-model
"""
from google.cloud import translate_v2 as translate

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./asset/fluid-mote-345206-ae29a133c3e7.json"
translate_client = translate.Client()

def translate_text_with_google(text: str, target='zh', model="nmt") -> str:
    global translate_client

    if isinstance(text, bytes):
        text = text.decode("utf-8")
    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language=target, model=model)
    return result["translatedText"]

def translate_text_with_google_batch(text: List[str], target='zh', model="nmt") -> List[str]:
    ret = Parallel(n_jobs=32, backend="threading")(delayed(translate_text_with_google)(t, target=target, model=model) for t in text)
    return ret
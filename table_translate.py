import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Set
import requests
from sentence_splitter import split_text_into_sentences
import json
from tqdm import tqdm
import argparse
import logging
import cohere
import re
import dataframe_image as dfi
import pandas as pd
import base64
import io
from PIL import Image



def inference_request(url: str, source_language: str, target_language: str, texts: List[str]) -> List[str]:

    headers = {"Content-Type": "application/json"}
    data = {
        "source_language": source_language,
        "target_language": target_language,
        "texts": texts,
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()["translated_texts"]


def call_inference_api(
    example: Dict[str, List[str]],
    url: str,
    source_language_code: str,
    target_language_code: str,
    keys_to_be_translated: List[str],
) -> Dict[str, List[str]]:

    for key in keys_to_be_translated:
        # NLLB model seems to ignore some sentences right before newline characters
        batch_str = [sen.replace('\n', '') for sen in example[key]]
        example[key] = inference_request(url, source_language_code, target_language_code, batch_str)
    return example


def translate_table(
    inputs
) -> Dict[str, List[str]]:

    table, url, source_language_code, target_language_code = inputs

    translated_table = []
    for row in table:
        translate_row = []
        for cell in row:
            if bool(re.fullmatch(r'^[\d\W_]*$', str(cell))):
                translate_row.append(cell)
            else:
                translation = {'text': cell}

                keys_to_be_translated = ["text"]

                sentenized_example = defaultdict(list)

                for k in keys_to_be_translated:
                    sentenized_example[f"{k}_pos"].append(0)

                for k in translation.keys():
                    sentences = split_text_into_sentences(text=translation[k], language='en')
                    sentenized_example[k].extend(sentences)
                    sentenized_example[f"{k}_pos"].append(sentenized_example[f"{k}_pos"][-1] + len(sentences))
                
                result = call_inference_api(example=sentenized_example,
                                            url=url,
                                            keys_to_be_translated=keys_to_be_translated,
                                            source_language_code=source_language_code,
                                            target_language_code=target_language_code)
                
                for k in keys_to_be_translated:
                    merged_texts = []
                    l = 0
                    r = 1
                    while r < len(result[f"{k}_pos"]):
                        start = result[f"{k}_pos"][l]
                        end = result[f"{k}_pos"][r]
                        merged_texts.append(' '.join(result[k][start:end]))
                        l += 1
                        r += 1
                    translation[k] = merged_texts[0]
                
                translate_row.append(translation['text'])

        translated_table.append(translate_row)

    return translated_table

def translate_dataset_via_inference_api(
    dataset_path,
    target_language_code: str,
    source_language_code: str,
    url: str = "http://localhost:8000/translate",
    output_dir: str = "./datasets",
) -> None:

    start_time = time.time()

    size = 0
    with open(dataset_path, "r") as file:
        dataset = [json.loads(line) for line in file]
    
    with open(output_dir, "w") as f:
        for i, data in enumerate(tqdm(dataset)):
            translated_table_images = []
            translated_tables = []
            # data = json.loads(line)
            tables = data['Table']
            for j, table in enumerate(tables):
                translated_table = translate_table((table, url, source_language_code, target_language_code))
                translated_tables.append(translated_table)
            data['Translated_Table'] = translated_tables
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            size += 1

    print(f"Translated {size} samples")

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    return translate_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate datasets from huggingface hub using a variety of methods.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset file")
    parser.add_argument("--source_language_code", type=str, help="Target language code")
    parser.add_argument("--target_language_code", type=str, help="Target language code")
    parser.add_argument("--url", type=str, default="http://localhost:8000/translate", help="URL of the inference API server")
    parser.add_argument("--output_dir", type=str, default="./datasets", help="Output directory for the translated dataset")
    args = parser.parse_args()
    print(args)

    translate_dataset_via_inference_api(
        dataset_path=args.dataset_path,
        source_language_code=args.source_language_code,
        target_language_code=args.target_language_code,
        url=args.url,
        output_dir=args.output_dir,
    )


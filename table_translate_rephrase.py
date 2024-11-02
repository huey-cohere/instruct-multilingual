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

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

HEADER_COLORS = ['lightgreen', 'green', 'lightsteelblue', 'powderblue', 'sandybrown', 'lightsalmon', 'lightskyblue', 'lightgray', 'greenyellow', 'lightseagreen', 'lightslategray', ]
BACKGROUND_COLORS = ['lightblue', 'aqua', 'cyan', 'honeydew', 'ivory', 'lemonchiffon', 'ghostwhite', 'gainsboro', 'mistyrose', 'powderblue', 'snow', 'whitesmoke', 'lime', 'lightskyblue','khaki', 'mediumaquamarine']  


client = cohere.ClientV2("R8fQH9pZzw70Ixq3eOmcLKiCaZVS0wHs7eUc82dU", base_url="https://stg.api.cohere.ai")

PROMPT =  """Original Text: 
{raw_text}\n

Translation: 
{translation}\n

Instruction:
Given the original text and its translation, improve the quality of the translation by rephrasing it. 
Ensure the rephrased translation closely aligns with the original text in meaning, structure, tone, and style. 
Make the rephrased translation sound natural and fluent in the target language while preserving the core message, correcting any grammatical errors, and retaining all stylistic elements (e.g., enumeration, punctuation, capitalization, spacing, line breaks, etc.) from the original.

The output must strictly follow this format:
Rephrased Translation: <rephrased translation placeholder>"""


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


def translate_sent_by_sent(
    inputs
) -> Dict[str, List[str]]:

    raw_text, url, source_language_code, target_language_code = inputs
    translation = {'text': raw_text}

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
    
    rephrase_input = PROMPT.format(raw_text=raw_text, translation=translation['text'])

    retry_count = 0
    response_user = None
    response_chatbot = None
    while retry_count < 30:
        try:
            response_user = client.chat(
                model='command-r-plus',
                messages=[
                    {
                        "role": "user",
                        "content": rephrase_input
                    }
                ],
                temperature = 0.5,
                p = 0.9,
                max_tokens = 1024,
            )
            rephrase_output = response_user.message.content[0].text.strip()
            match_rephrase_output = re.search(r'Rephrased Translation[:：∶﹕]([\s\S]*)', rephrase_output)
            if match_rephrase_output:
                rephrase_output_extract = match_rephrase_output.group(1).strip()
            else:
                raise Exception("No match found")
            
            return rephrase_output_extract
        
        except Exception as e:
            # print(f"API Error: {e}")
            # print(f"Retry count: {retry_count}")
            # print("Retrying in 10 seconds")
            logging.error(f"API Error: {e}")
            logging.error(f"Retry count: {retry_count}")
            logging.error("Retrying in 3 seconds")
            if retry_count == 28:
                logging.error(f"Failed: {response_user}")
                logging.error(f"Failed: {response_chatbot}")
            time.sleep(3)
            retry_count += 1
    
    return None



def translate_dataset_via_inference_api(
    dataset,
    target_language_code: str,
    source_language_code: str,
    url: str = "http://localhost:8000/translate",
    output_dir: str = "./datasets",
) -> None:

    start_time = time.time()

    # translated_dataset = []
    
    # for data in tqdm(dataset):
    #     translated_dataset.append(translate_sent_by_sent((data, url, source_language_code, target_language_code)))

    translate_table = []
    for row in data:
        translate_row = []
        for cell in row:   
            if not bool(re.fullmatch(r'^[\d\W_]*$', cell)):
                # translate_cell = translate_text(cell, 'zh-CN')
                translate_cell = translate_sent_by_sent((cell, url, source_language_code, target_language_code))
                translate_row.append(translate_cell)
            else:
                translate_row.append(cell)
        translate_table.append(translate_row)
    
                           
    # print(f"Translated {len(translated_dataset)} samples")

    # with open(output_dir, "w") as f:
    #     for data in translated_dataset:
    #         f.write(json.dumps(data, ensure_ascii=False) + "\n")

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    return translate_table

def translate_dataset(url: str = "http://localhost:8000/translate",
                      output_dir: str = "./datasets",) -> None:

    with open('table_data/finqa_train.json') as f:
        finqa = json.load(f)
    
    print(f"Dataset size: {len(finqa)}")

    translate_dataset_via_inference_api(
        dataset=finqa,
        source_language_code=source_language_code,
        target_language_code=target_language_code,
        url=url,
        output_dir=output_dir,
    )

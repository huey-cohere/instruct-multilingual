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


def covert_to_table_image(table):

    df = pd.DataFrame(table)

    styled_df = (
        df.style
        .hide(axis="index")
        .hide(axis="columns")
        .set_table_styles([
            {'selector': 'tbody tr:nth-child(n+2)', 'props': [('background-color', random.choice(BACKGROUND_COLORS))]},
            {'selector': 'tbody tr:nth-child(1)', 'props': [('background-color', random.choice(HEADER_COLORS))]},
            {'selector': 'table', 'props': [
                ('border', '1px solid white'),
            ]},
            {'selector': 'td', 'props': [
                ('min-width', '150px'), 
                ('max-width', '450px'),
                ('padding', '15px'),
            ]}
        ])
        .set_properties(**{
            'text-align': 'center',
            'font-size': '12px',
        })
    )

    # dfi.export(styled_df,f"images/{name}.jpeg")
    buffer = io.BytesIO()
    dfi.export(styled_df, buffer, format="jpeg")
    buffer.seek(0)  # Move to the start of the buffer

    # with open(f'images/{name}.jpeg', "rb") as image_file:
    #     encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    encoded_image = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()

    return f"data:image/jpeg;base64,{encoded_image}"

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

def rephrase(raw_text, translation, engine, temperature, top_p, max_tokens):

    rephrase_input = PROMPT.format(raw_text=raw_text, translation=translation)

    retry_count = 0
    response_output = None
    while retry_count < 30:
        try:
            response_output = client.chat(
                model=engine,
                messages=[
                    {
                        "role": "user",
                        "content": rephrase_input
                    }
                ],
                temperature = temperature,
                p = top_p,
                max_tokens = max_tokens,
            )
            rephrase_output = response_output.message.content[0].text.strip()
            match_rephrase_output = re.search(r'Rephrased Translation[:：∶﹕]([\s\S]*)', rephrase_output)
            if match_rephrase_output:
                rephrase_output_extract = match_rephrase_output.group(1).strip()
            else:
                raise Exception("No match found")
            
            return rephrase_output_extract
        
        except Exception as e:
            logging.error(f"API Error: {e}")
            logging.error(f"Retry count: {retry_count}")
            logging.error("Retrying in 3 seconds")
            if retry_count == 28:
                logging.error(f"Failed: {response_output}")
            time.sleep(3)
            retry_count += 1
    
    return None

def translate_table(
    inputs
) -> Dict[str, List[str]]:

    table, url, engine, source_language_code, target_language_code, max_tokens, temperature, top_p = inputs

    translated_table = []
    for row in table:
        translate_row = []
        for cell in row:
            if bool(re.fullmatch(r'^[\d\W_]*$', cell)):
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
                
                rephrased_translation = rephrase(cell, translation['text'], engine, temperature, top_p, max_tokens)

                if rephrased_translation is not None:
                    translate_row.append(rephrased_translation)
                else:
                    translate_row.append(cell)

        translated_table.append(translate_row)

                

def translate_dataset_via_inference_api(
    dataset_path,
    target_language_code: str,
    source_language_code: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    engine: str,
    url: str = "http://localhost:8000/translate",
    output_dir: str = "./datasets",
) -> None:

    start_time = time.time()

    
    size = 0
    with open(dataset_path, "r") as file, open(output_dir, "w") as f:
        for line in file:
            translated_table_images = []
            translated_tables = []
            data = json.loads(line)
            tables = data['tables']
            for table in tables:
                translated_table = translate_table((table, url, engine, source_language_code, target_language_code, max_tokens, temperature, top_p))
                translated_tables.append(translated_table)
                translated_table_images.append(covert_to_table_image(translated_table))
            data['Translated_Image'] = translated_table_images
            data['Translated_Table'] = translated_tables
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            size += 1

    # for data in dataset:
    #     tables = data['tables']
    #     for table in tables:
    #         translated_table = translate_table((table, url, engine, source_language_code, target_language_code, max_tokens, temperature, top_p))

                           
    # print(f"Translated {len(translated_dataset)} samples")

    # with open(output_dir, "w") as f:
    #     for data in translated_dataset:
    #         f.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"Translated {size} samples")

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    return translate_table

# def translate_dataset(dataset_path,
#                       source_language_code: str,
#                       target_language_code: str,
#                       max_tokens: int,
#                       temperature: float,
#                       top_p: float,
#                       engine: str,
#                       url: str = "http://localhost:8000/translate",
#                       output_dir: str = "./datasets",) -> None:

#     with open(dataset_path, "r") as file:
#         # dataset = []
#         # for line in file:
#         #     dataset.append(json.loads(line))
#         #     if len(dataset) == 10:
#         #         break
#         dataset = [json.loads(line) for line in file]
#     # print(dataset[0])

#     print(f"Dataset size: {len(dataset)}")

#     translate_dataset_via_inference_api(
#         dataset=dataset,
#         source_language_code=source_language_code,
#         target_language_code=target_language_code,
#         url=url,
#         output_dir=output_dir,
#         max_tokens=max_tokens,
#         temperature=temperature,
#         top_p=top_p,
#         engine=engine,
#     )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate datasets from huggingface hub using a variety of methods.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset file")
    parser.add_argument("--source_language_code", type=str, help="Target language code")
    parser.add_argument("--target_language_code", type=str, help="Target language code")
    parser.add_argument("--url", type=str, default="http://localhost:8000/translate", help="URL of the inference API server")
    parser.add_argument("--output_dir", type=str, default="./datasets", help="Output directory for the translated dataset")
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature for sampling",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top p for sampling",
    )
    args = parser.parse_args()
    print(args)
    # translate_dataset(
    #     dataset_path=args.dataset_path,
    #     target_language_code=args.target_language_code,
    #     source_language_code=args.source_language_code,
    #     url=args.url,
    #     output_dir=args.output_dir,
    #     engine = 'command-r-plus',
    #     max_tokens=args.max_tokens,
    #     temperature=args.temperature,
    #     top_p=args.top_p,
    # )
    translate_dataset_via_inference_api(
        dataset_path=args.dataset_path,
        source_language_code=args.source_language_code,
        target_language_code=args.target_language_code,
        url=args.url,
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        engine='command-r-plus',
    )


"""Translate datasets from huggingface hub using a variety of methods."""
import os
import random
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Set
import requests
from sentence_splitter import split_text_into_sentences
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import datasets

# aya23_code2lang ={
#     "arb_Arab": "Modern Standard Arabic",
#     "zho_Hans": "Chinese (Simplified)",
#     "zho_Hant": "Chinese (Traditional)",
#     "ces_Latn": "Czech",
#     "nld_Latn": "Dutch",
#     "eng_Latn": "English",
#     "fra_Latn": "French",
#     "deu_Latn": "German",
#     "ell_Grek": "Greek",
#     "heb_Hebr": "Hebrew",
#     "hin_Deva": "Hindi",
#     "ind_Latn": "Indonesian",
#     "ita_Latn": "Italian",
#     "jpn_Jpan": "Japanese",
#     "kor_Hang": "Korean",
#     "pes_Arab": "Western Persian",
#     "pol_Latn": "Polish",
#     "por_Latn": "Portuguese",
#     "ron_Latn": "Romanian",
#     "rus_Cyrl": "Russian",
#     "spa_Latn": "Spanish",
#     "tur_Latn": "Turkish",
#     "ukr_Cyrl": "Ukrainian",
#     "vie_Latn": "Vietnamese",
# }

# aya23_lang2code = {v: k for k, v in aya23_code2lang.items()}

def inference_request(url: str, source_language: str, target_language: str, texts: List[str]) -> List[str]:
    """A HTTP POST request to the inference server for translation.

    Args:
        url (str): The URL of the inference API server
        source_language (str): Language code for the source language
        target_language (str): Language code for the target language
        texts (List[str]): List of texts to be translated

    Returns:
        List[str]: List of translated text
    """

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
    """Calls the inference server key-by-key and returns the translated
    examples back.

    Args:
        example (Dict[str,List[str]]): A batch of inputs from the dataset for translation. Keys are the column names, values are the batch of text inputs
        url (str): The URL of the inference API server.
        source_lang_code (str): Language code for the source language
        target_lang_code (str): Language code for the target language
        keys_to_be_translated (List[str], optional): The keys/columns for the texts you want translated. Defaults to ["dialogue", "summary"].

    Returns:
        Dict[str,List[str]]: Translated outputs based on the example Dict
    """
    for key in keys_to_be_translated:
        # NLLB model seems to ignore some sentences right before newline characters
        batch_str = [sen.replace('\n', '') for sen in example[key]]
        example[key] = inference_request(url, source_language_code, target_language_code, batch_str)
    return example



def translate_sent_by_sent(
    inputs
) -> Dict[str, List[str]]:
    """A wrapper for call_inference_api that preprocess the input text by
    breaking them into sentences.

    Args:
        example (Dict[str,List[str]]): A batch of inputs from the dataset for translation. Keys are the column names, values are the batch of text inputs
        url (str): The URL of the inference API server.
        source_lang_code (str): Language code for the source language
        target_lang_code (str): Language code for the target language
    Returns:
        Dict[str,List[str]]: Translated outputs based on the example Dict
    """
    data_dict, url, source_language_code, target_language_code = inputs
    example = {}

    # for turn in data["turns"]:
    #     if turn["role"] == "User":
    #         for content in turn['content']:
    #             if "text" in content:
    #                 example['question'] = content["text"]
    #     if turn["role"] == "Chatbot":
    #         example['answer'] = turn["content"]

    for data in data_dict["User"]:
        if data['language'] == source_language_code and data['source'] == "raw-processed":
            example["User"] = data["text"]
            break
    
    for data in data_dict["Chatbot"]:
        if data['language'] == source_language_code and data['source'] == "raw-gpt_recap":
            example["Chatbot"] = data["text"]
            break

    if example == {}:
        raise ValueError("No data found for translation")

    keys_to_be_translated = ["User", "Chatbot"]

    sentenized_example = defaultdict(list)

    for k in keys_to_be_translated:
        sentenized_example[f"{k}_pos"].append(0)

    for k in example.keys():
        sentences = split_text_into_sentences(text=example[k], language='en')
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
        example[k] = merged_texts[0]

    # for i, _ in enumerate(data["turns"]):
    #     if data["turns"][i]["role"] == "User":
    #         for j, _ in enumerate(data["turns"][i]["content"]):
    #             if "text" in data["turns"][i]["content"][j]:
    #                 data["turns"][i]["content"][j]['text'] = example["question"]
        
    #     if data["turns"][i]["role"] == "Chatbot":
    #         data["turns"][i]["content"] = example["answer"]

    data_dict["User"].append({"text": example["User"], "language": target_language_code, "source": "raw-processed-nllb_translated"})
    data_dict["Chatbot"].append({"text": example["Chatbot"], "language": target_language_code, "source": "raw-gpt_recap-nllb_translated"})

    return data_dict




# def translate_batch(inputs):
#     batch, url, target_lang_code = inputs
#     return [
#         translate_sent_by_sent(
#             item,
#             url=url,
#             target_lang_code=target_lang_code
#         )
#         for item in batch
#     ]

def translate_dataset_via_inference_api(
    dataset_path,
    target_language_code: str,
    source_language_code: str,
    url: str = "http://localhost:8000/translate",
    output_dir: str = "./datasets",
) -> None:

    start_time = time.time()
    size = 0
    with open(dataset_path, "r") as file, open(output_dir, "w") as output_file:
        for line in tqdm(file):
            data = json.loads(line)
            translated_data = translate_sent_by_sent((data, url, source_language_code, target_language_code))
            output_file.write(json.dumps(translated_data, ensure_ascii=False) + "\n")
            size += 1

    # translated_dataset = []

    # with ThreadPoolExecutor(max_workers=4) as executor:
    #     # futures = []
    #     # for i in range(0, len(dataset), 10000):
    #     #     batch = dataset[i:i + 10000]
    #     #     futures.append(executor.submit(translate_batch, (batch, url, target_language_code)))
        
    #     futures = [
    #         executor.submit(
    #             translate_sent_by_sent, (data, url, target_language_code)
    #         )
    #         for data in dataset
    #     ]
            
    #     for future in tqdm(as_completed(futures), total=len(futures)):
    #         result = future.result() 
    #         if result is not None:
    #             translated_dataset.append(result)

    
    # for data in tqdm(dataset):
    #     translated_dataset.append(translate_sent_by_sent((data, url, source_language_code, target_language_code)))


    # print(f"Translated {len(translated_dataset)} samples")
    
    # # print(translated_dataset)

    # # os.makedirs(output_dir, exist_ok=True)
    # with open(output_dir, "w") as f:
    #     for data in translated_dataset:
    #         f.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"Translated {size} samples")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Elapsed time: {elapsed_time:.4f} seconds")


# def translate_dataset(dataset_path,
#                       source_language_code: str,
#                       target_language_code: str,
#                       url: str = "http://localhost:8000/translate",
#                       output_dir: str = "./datasets",) -> None:

#     # with open(dataset_path, "r") as file:
#     #     # dataset = []
#     #     # for line in file:
#     #     #     dataset.append(json.loads(line))
#     #     #     if len(dataset) == 10:
#     #     #         break
#     #     dataset = [json.loads(line) for line in file]
#     # # print(dataset[0])

#     # print(f"Dataset size: {len(dataset)}")

#     # for code, language in aya23_code2lang.items():
#     #     if language != "English":
#     #         print(f"Currently translating: {language}")
#     #         translate_dataset_via_inference_api(
#     #             dataset=dataset,
#     #             target_language_code=code,
#     #             url=url,
#     #             output_dir=output_dir,
#     #         )

#     translate_dataset_via_inference_api(
#         dataset_path=dataset_path,
#         source_language_code=source_language_code,
#         target_language_code=target_language_code,
#         url=url,
#         output_dir=output_dir,
#     )

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
        target_language_code=args.target_language_code,
        source_language_code=args.source_language_code,
        url=args.url,
        output_dir=args.output_dir,
    )


import abc
import asyncio
from dataclasses import dataclass
from typing import Optional, Union
from tqdm import tqdm
import re
import os
from cohere import AsyncClient
import argparse
import pandas as pd
from langcodes import Language

COHERE_API_KEY = 'XvksMYmZbQF0aagROSgmbTbiL4auprJmQtWk7iFP'

PROMPT =  """Original Text: 
{raw_text}\n

Translation: 
{translation}\n

Instruction:
Given the original text and its translation, improve the quality of the translation by rephrasing it. 
Ensure the rephrased translation closely aligns with the original text in meaning, structure, tone, and style. 
Make the rephrased translation sound natural and fluent in the target language ({language}) while preserving all essential details, correcting any grammatical errors, and retaining all stylistic elements (e.g., enumeration, parentheses, punctuation, capitalization, spacing, line breaks, etc.) from the original.

The output must strictly follow this format:
Rephrased Translation: <rephrased translation placeholder>"""


class asyncio_rephrase_table(abc.ABC):
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        engine: str = "command-r-plus",
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_completion_tokens: int = 1024,
        max_workers = 10,
    ):

        self.base_url = base_url
        self.engine= engine
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.max_completion_tokens = max_completion_tokens
        self.max_workers = max_workers
        self.client = self.make_client()

    def make_client(self):
        return AsyncClient(api_key=self.api_key, base_url=self.base_url)

    def get_client(self):
        return self.client

    async def _get_completion_from_client(
        self,
        message,
    ):
        num_retries = 0
        while num_retries <= 30:
            try:
                response = await self.client.chat(
                    chat_history = None,
                    message=message,
                    model=self.engine,
                    temperature=self.temperature,
                    p=self.top_p,
                    max_tokens=self.max_completion_tokens,
                )
            
                output = response.text.strip()
                match_output = re.search(r'Rephrased Translation[:：∶﹕]([\s\S]*)', output)
                if match_output:
                    extract_output = match_output.group(1).strip()
                    return extract_output
                else:
                    raise Exception("No match found")

            except Exception as e:
                print(f"API Error: {e}")
                print(f"Retry count: {num_retries}")
                print("Retrying in 3 seconds")
                await asyncio.sleep(3)
                num_retries += 1

        return None
                
    async def _get_completion_async(
        self,
        data_dict,
        language_code,
    ):
        
        for user in data_dict['User']:
            if user['language'] == "eng_Latn" and user['source'] == "raw-processed":
                raw_user = user['text']
            if user['language'] == target_language_code and user['source'] == "raw-processed-nllb_translated":
                translated_user = user['text']
        
        for bot in data_dict['Chatbot']:
            if bot['language'] == "eng_Latn" and bot['source'] == "raw-gpt_recap":
                raw_chatbot = bot['text']
            if bot['language'] == target_language_code and bot['source'] == "raw-gpt_recap-nllb_translated":
                translated_chatbot = bot['text']
        
        language = Language.get(language_code).display_name()
        formatted_input_user = PROMPT.format(raw_text=raw_user, translation=translated_user, language=language)
        formatted_input_chatbot = PROMPT.format(raw_text=raw_chatbot, translation=translated_chatbot, language=language)

        response_user = await self._get_completion_from_client(
            message = formatted_input_user,
        )

        data_dict['User'].append(
                {
                    "text": response_user if response_user else raw_user,
                    "language": language_code,
                    "source": "raw-processed-nllb_translated-command_r_rephrase",
                }
        )

        response_chatbot = await self._get_completion_from_client(
            message = formatted_input_chatbot,
        )

        data_dict['Chatbot'].append(
                {
                    "text": response_chatbot if response_chatbot else raw_chatbot,
                    "language": language_code,
                    "source": "raw-gpt_recap-nllb_translated-command_r_rephrase",
                }
        )

        return data_dict


    def get_batch_completions(
        self,
        dataset
        language_code,
    ):

        async def process_batch():
            semaphore = asyncio.Semaphore(self.max_workers)

            async def bounded_get_completion(data_dict, index):
                async with semaphore:
                    result = await self._get_completion_async(
                        data_dict= data_dict,
                        language_code = language_code,
                    )
                    return index, result

            tasks = [bounded_get_completion(data_dict, i) for i, (data_dict)  in enumerate(dataset)]

            completions = [None] * len(raw_tables_all)
            for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                index, result = await task
                completions[index] = result
            return completions

        return self.run_on_event_loop(process_batch())
    

    def run_on_event_loop(self, async_fn):

        loop = asyncio.new_event_loop()

        output = loop.run_until_complete(async_fn)
        loop.run_until_complete(self.close_client())

        loop.close()

        self.client = self.make_client()

        return output


    async def close_client(self):
        if isinstance(self.client, list):
            for client in self.client:
                if hasattr(client, "close"):
                    await client.close()
        else:
            if hasattr(self.client, "close"):
                await self.client.close()

def main(args):

    provider = asyncio_rephrase_table(
        api_key=COHERE_API_KEY, 
        temperature=0.5, 
        top_p=0.9, 
        max_completion_tokens=2048, 
        engine='command-r-plus', 
        base_url="https://stg.api.cohere.com",
        max_workers = 96,
        )

    for language_code in os.listdir(args.dataset_path):

        print(f"Rephrasing {language_code}")

        df = pd.read_json(os.path.join(args.dataset_path, lang, "train.jsonl"), lines=True)
    
        completions = provider.get_batch_completions(
            dataset = dataset,
            language_code = language_code,
        )

        
        # completions = [c[0] for c in completions]

        # df = pd.read_json(os.path.join(args.dataset_path, lang, "train.jsonl"), lines=True)

        # df['Translated_Table'] = completions
        
        # os.makedirs(os.path.join(args.output_dir, lang), exist_ok=True)
        # with open(os.path.join(args.output_dir, lang, "train.jsonl"), 'w', encoding='utf-8') as file:
        #     df.to_json(file, orient="records", lines=True,  force_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate datasets from huggingface hub using a variety of methods.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset file", default='/home/olivernan_cohere_com/recap_data_translation_2024_11_01_raw_repharsed_table_translated/RecapMultiHiertt_translation')
    parser.add_argument("--output_dir", type=str, default='/home/olivernan_cohere_com/recap_data_translation_2024_11_01_raw_repharsed_table_translated_rephrase/RecapMultiHiertt_translation', help="Output directory for the translated dataset")
    args = parser.parse_args()
    print(args)
    main(args)






# import argparse
# from tqdm import tqdm
# import os
# import time
# import concurrent.futures
# import json
# import io
# import cohere
# import re
# import functools
# import gcsfs
# import random
# import logging
# logging.basicConfig(
#     level=logging.ERROR,
#     format="%(asctime)s - %(levelname)s - %(message)s",
# )
# @functools.lru_cache(maxsize=1)
# def gcfs():
#     return gcsfs.GCSFileSystem()

# # KEYS = [
# #     "XvksMYmZbQF0aagROSgmbTbiL4auprJmQtWk7iFP",
# #     "lTYH3NXmWOIHN3YTlTeqKpU4P9KE6OCPZVloFl1r",
# #     "d2sp1KIdlgtZLFNNtsuKyjeVCOhNftX5JrhSqc0w",
# #     "uYmBYC5XvWkZOqGVDmDHXNruifzdGX4orSqNKXqk",
# #     "tSXdF7KPl9ZZtxo3ZnVn3DTtCTOFd5InbyGsjrJM",
# #     ]

# client = cohere.ClientV2("tSXdF7KPl9ZZtxo3ZnVn3DTtCTOFd5InbyGsjrJM", base_url="https://stg.api.cohere.com") # base_url="https://stg.api.cohere.ai"


# PROMPT =  """Original Text: 
# {raw_text}\n

# Translation: 
# {translation}\n

# Instruction:
# Given the original text and its translation, improve the quality of the translation by rephrasing it. 
# Ensure the rephrased translation closely aligns with the original text in meaning, structure, tone, and style. 
# Make the rephrased translation sound natural and fluent in the target language ({language}) while preserving all essential details, correcting any grammatical errors, and retaining all stylistic elements (e.g., enumeration, parentheses, punctuation, capitalization, spacing, line breaks, etc.) from the original.

# The output must strictly follow this format:
# Rephrased Translation: <rephrased translation placeholder>"""

# def make_request(params):

#     data_dict, engine, target_language_code, max_tokens, temperature, top_p = params
    
#     for user in data_dict['User']:
#         if user['language'] == "eng_Latn" and user['source'] == "raw-processed":
#             raw_user = user['text']
#         if user['language'] == target_language_code and user['source'] == "raw-processed-nllb_translated":
#             translated_user = user['text']
    
#     for bot in data_dict['Chatbot']:
#         if bot['language'] == "eng_Latn" and bot['source'] == "raw-gpt_recap":
#             raw_chatbot = bot['text']
#         if bot['language'] == target_language_code and bot['source'] == "raw-gpt_recap-nllb_translated":
#             translated_chatbot = bot['text']

#     formatted_input_user = PROMPT.format(raw_text=raw_user, translation=translated_user)
#     formatted_input_chatbot = PROMPT.format(raw_text=raw_chatbot, translation=translated_chatbot)

#     retry_count = 0
#     response_user = None
#     response_chatbot = None
#     while retry_count < 30:
#         try:
#             response_user = client.chat(
#                 model=engine,
#                 messages=[
#                     {
#                         "role": "user",
#                         "content": formatted_input_user
#                     }
#                 ],
#                 temperature = temperature,
#                 p = top_p,
#                 max_tokens = max_tokens,
#             )
#             output_user = response_user.message.content[0].text.strip()
#             match_user = re.search(r'Rephrased Translation[:：∶﹕]([\s\S]*)', output_user)
#             if match_user:
#                 response_user_extract = match_user.group(1).strip()
#             else:
#                 raise Exception("No match found")
            
#             data_dict['User'].append(
#                 {
#                     "text": response_user_extract,
#                     "language": target_language_code,
#                     "source": "raw-processed-nllb_translated-command_r_rephrase",
#                 }
#             )
            
#             response_chatbot = client.chat(
#                 model=engine,
#                 messages=[
#                     {
#                         "role": "user",
#                         "content": formatted_input_chatbot
#                     }
#                 ],
#                 temperature = temperature,
#                 p = top_p,
#                 max_tokens = max_tokens,
#             )
#             output_chatbot = response_chatbot.message.content[0].text.strip()
#             match_chatbot = re.search(r'Rephrased Translation[:：∶﹕]([\s\S]*)', output_chatbot)
#             if match_chatbot:
#                 response_chatbot_extract = match_chatbot.group(1).strip()
#             else:
#                 raise Exception("No match found")

#             data_dict['Chatbot'].append(
#                 {
#                     "text": response_chatbot_extract,
#                     "language": target_language_code,
#                     "source": "raw-gpt_recap-nllb_translated-command_r_rephrase",
#                 }
#             )

#             return data_dict
        
#         except Exception as e:
#             # print(f"API Error: {e}")
#             # print(f"Retry count: {retry_count}")
#             # print("Retrying in 10 seconds")
#             logging.error(f"API Error: {e}")
#             logging.error(f"Retry count: {retry_count}")
#             logging.error("Retrying in 3 seconds")
#             if retry_count == 28:
#                 logging.error(f"Failed: {response_user}")
#                 logging.error(f"Failed: {response_chatbot}")
#             time.sleep(3)
#             retry_count += 1
    
#     return None


# def run_query(
#     num_threads,
#     output_dir,
#     dataset_path,
#     engine,
#     max_tokens,
#     temperature,
#     top_p,
#     target_language_code,
# ):
#     # with gcfs().open(dataset_path, "r") as file:
#     with open(dataset_path, "r") as file:
#         dataset = [json.loads(line) for line in file]

#     # dataset = dataset[:10]
#     print(f"dataset size: {len(dataset)}")
#     # print(dataset[0])
#     # print(dataset[1])

#     with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:

#         futures = [
#             executor.submit(
#                 make_request, (data, engine, target_language_code, max_tokens, temperature, top_p)
#             )
#             for data in dataset
#         ]

#         new_dataset = []
#         for future in tqdm(
#             concurrent.futures.as_completed(futures), total=len(futures)
#         ):
#             result = future.result()
#             if result is not None:
#                 new_dataset.append(result)

#     print(new_dataset[0])
#     print(f"new dataset size: {len(new_dataset)}")

#     with open(output_dir, "w+") as f:
#         for data in new_dataset:
#             f.write(json.dumps(data, ensure_ascii=False) + "\n")


# if __name__ == "__main__":

#     parser = argparse.ArgumentParser(description="Run generation on a dataset")
#     parser.add_argument(
#         "--dataset_path",
#         type=str,
#         default=None,
#         help="Path to the dataset",
#     )
#     parser.add_argument(
#         "--num_threads",
#         type=int,
#         default=48,
#         help="Number of threads to use for preprocessing",
#     )
#     parser.add_argument(
#         "--output_dir",
#         type=str,
#         default="/home/olivernan/",
#         help="Output directory for generated data",
#     )
#     parser.add_argument(
#         "--max_tokens",
#         type=int,
#         default=1024,
#         help="Maximum number of tokens to generate",
#     )
#     parser.add_argument(
#         "--temperature",
#         type=float,
#         default=0.5,
#         help="Temperature for sampling",
#     )
#     parser.add_argument(
#         "--top_p",
#         type=float,
#         default=0.8,
#         help="Top p for sampling",
#     )
#     parser.add_argument(
#         "--target_language_code",
#         type=str
#     )

#     args = parser.parse_args()

#     print(args)

#     # run_inference(num_proc=args.num_proc, output_dir=args.output_dir, dataset_path=args.dataset_path)

#     run_query(
#         num_threads=args.num_threads,
#         output_dir=args.output_dir,
#         dataset_path=args.dataset_path,
#         engine='command-r-plus',
#         max_tokens=args.max_tokens,
#         temperature=args.temperature,
#         top_p=args.top_p,
#         target_language_code=args.target_language_code,
#     )

#     # python gpt_recaption.py --dataset_path /home/olivernan_cohere_com/gpt-recaption/CauldronAi2d/raw_data/train.jsonl --num_threads 64 --output_dir /home/olivernan_cohere_com/gpt-recaption/CauldronAi2d/2024_09_09/train.jsonl --max_tokens 512 --temperature 1.0 --top_p 0.95 --n 1





import yaml
import ast
import os
from datetime import date
import argparse
from google.cloud import storage
import subprocess
import json
import shutil
# import functools
# import gcsfs

# @functools.lru_cache(maxsize=1)
# def gcfs():
#     return gcsfs.GCSFileSystem()

DATA_DIR = "/home/olivernan_cohere_com/recap_data_translation_2024_11_01_raw"

OUTPUT_DIR = "/home/olivernan_cohere_com/recap_data_translation_2024_11_01_raw_rephrased"
os.makedirs(OUTPUT_DIR, exist_ok=True)

dataset_name_list = [
                    # 'RecapCauldronAokvqa_translation', 
                    # 'RecapCauldronClevr_translation', 
                    # 'RecapCauldronCocoqa_translation', 
                    # 'RecapCauldronGeomverse_translation', 
                    # 'RecapCauldronIconqa_translation', 
                    # 'RecapCauldronLocalized_narratives_translation',
                    # 'RecapCauldronNlvr2_translation', 
                    # 'RecapCauldronRaven_translation', 
                    # 'RecapCauldronSpot_the_diff_translation', 
                    # 'RecapCauldronTallyqa_translation',
                    # 'RecapCauldronVqarad_translation',
                    # 'RecapCauldronVqav2_translation', 
                    # 'RecapCauldronVsr_translation',
                    # 'RecapClevrMath_translation',
                    # 'RecapVisual7Ws_translation'
                    # 'RecapFINQA_translation',
                    'RecapMultiHiertt_translation',
                    ]

def main(
    max_tokens: int,
    temperature: float,
    top_p: float,
    num_threads: int,
):
    os.makedirs("logs-2/", exist_ok=True)

    # for dataset_name in gcfs().listdir(DATA_DIR):
    for dataset_name in dataset_name_list: #os.listdir(DATA_DIR):
            
        print(f"Dataset: {dataset_name}")
        # processes = []
        # for language in gcfs().listdir(f"{DATA_DIR}/{dataset_name}"):
        for language in os.listdir(f"{DATA_DIR}/{dataset_name}"):

            # if language not in ['pol_Latn']:
            #     continue
            
            print(f"Dataset: {dataset_name}, Language: {language}")

            dataset_path = f"{DATA_DIR}/{dataset_name}/{language}/train.jsonl"

            output_folder = f"{OUTPUT_DIR}/{dataset_name}/{language}"
            os.makedirs(output_folder, exist_ok=True)

            command = f"""
                nohup python cauldron_nllb_translate_rephrase.py \
                --dataset_path {dataset_path} \
                --target_language_code {language} \
                --num_threads {num_threads} \
                --output_dir {output_folder}/train.jsonl \
                --max_tokens {max_tokens} \
                --temperature {temperature} \
                --top_p {top_p} > logs-2/{dataset_name}_{language}.out
                """

            print("Running command")
            print(command)
            process = subprocess.Popen(command, shell=True)
            process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="run command r repharse on cauldron datasets"
    )

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
    parser.add_argument(
        "--num_threads",
        type=int,
        default=64,
        help="Number of threads to use for preprocessing",
    )
    args = parser.parse_args()

    print(args)
    main(
        args.max_tokens,
        args.temperature,
        args.top_p,
        args.num_threads,
    )


















# import yaml
# import ast
# import os
# from datetime import date
# import argparse
# from google.cloud import storage
# import subprocess
# import json
# import shutil
# import math
# # import functools
# # import gcsfs

# # @functools.lru_cache(maxsize=1)
# # def gcfs():
# #     return gcsfs.GCSFileSystem()

# DATA_DIR = "/home/olivernan_cohere_com/recap_data_translation_2024_11_01_raw"

# OUTPUT_DIR = "/home/olivernan_cohere_com/recap_data_translation_2024_11_01_raw_rephrased"

# KEYS = [
#     "XvksMYmZbQF0aagROSgmbTbiL4auprJmQtWk7iFP",
#     "lTYH3NXmWOIHN3YTlTeqKpU4P9KE6OCPZVloFl1r",
#     "d2sp1KIdlgtZLFNNtsuKyjeVCOhNftX5JrhSqc0w",
#     "uYmBYC5XvWkZOqGVDmDHXNruifzdGX4orSqNKXqk",
#     "tSXdF7KPl9ZZtxo3ZnVn3DTtCTOFd5InbyGsjrJM",
#     ]
# dataset_name_list = [
#                     # 'RecapCauldronAokvqa_translation', 
#                     # 'RecapCauldronClevr_translation', 
#                     # 'RecapCauldronCocoqa_translation', 
#                     # 'RecapCauldronGeomverse_translation', 
#                     # 'RecapCauldronIconqa_translation', 
#                     # 'RecapCauldronLocalized_narratives_translation',
#                     # 'RecapCauldronNlvr2_translation', 
#                     # 'RecapCauldronRaven_translation', 
#                     # 'RecapCauldronSpot_the_diff_translation', 
#                     # 'RecapCauldronTallyqa_translation',
#                     # 'RecapCauldronVqarad_translation',
#                     # 'RecapCauldronVqav2_translation', 
#                     # 'RecapCauldronVsr_translation',
#                     # 'RecapClevrMath_translation',
#                     # 'RecapVisual7Ws_translation'
#                     # 'RecapFINQA_translation',
#                     'RecapMultiHiertt_translation',
#                     ]

# def main(
#     max_tokens: int,
#     temperature: float,
#     top_p: float,
#     num_threads: int,
# ):
#     os.makedirs("logs-2/", exist_ok=True)

#     # for dataset_name in gcfs().listdir(DATA_DIR):
#     for dataset_name in dataset_name_list: #os.listdir(DATA_DIR):
            
#         print(f"Dataset: {dataset_name}")
#         processes = []
#         # for language in gcfs().listdir(f"{DATA_DIR}/{dataset_name}"):
#         for language in os.listdir(f"{DATA_DIR}/{dataset_name}"):
            
#             # if language not in ['ita_Latn']:
#             #     continue

#             with open(os.path.join(DATA_DIR, dataset_name, language, 'train.jsonl'), "r") as file:
#                 dataset = [json.loads(line) for line in file]
            
#             num_per_chunk = math.ceil(len(dataset) / 5)
#             dataset_chunk = [
#                 dataset[i : i + num_per_chunk] for i in range(0, len(dataset), num_per_chunk)
#             ]
#             dataset_paths = []
#             os.makedirs(f"{OUTPUT_DIR}/{dataset_name}/{language}/raw_data/", exist_ok=True)
#             for i, chunk in enumerate(dataset_chunk):
#                 _path = f"{OUTPUT_DIR}/{dataset_name}/{language}/raw_data/split_{i}.jsonl"
#                 dataset_paths.append(_path)
#                 with open(_path, "w+") as f:
#                     for i, line in enumerate(chunk):
#                         f.write(json.dumps(line, ensure_ascii=False) + "\n")
            
#             print(f"Dataset: {dataset_name}, Language: {language}")

#             del dataset_chunk
#             del dataset

#             output_folder = f"{OUTPUT_DIR}/{dataset_name}/{language}/rephrased"
#             os.makedirs(output_folder, exist_ok=True)

#             for i, path in enumerate(dataset_paths):
                
#                 command = f"""
#                     nohup python cauldron_nllb_translate_rephrase.py \
#                     --dataset_path {path} \
#                     --target_language_code {language} \
#                     --num_threads {num_threads} \
#                     --output_dir {output_folder}/split_{i}.jsonl \
#                     --max_tokens {max_tokens} \
#                     --temperature {temperature} \
#                     --top_p {top_p} > logs-2/{dataset_name}_{language}_{i}.out
#                     """
#                 print("Running generation command")
#                 print(command)
#                 process = subprocess.Popen(command, shell=True)

#                 processes.append(process)

#                 if len(processes) == 5:
#                     for process in processes:
#                         process.wait()
#                     processes = [] 
            
#             for process in processes:
#                 process.wait()

#             count = 0
#             with open(
#                 f"{OUTPUT_DIR}/{dataset_name}/{language}/train.jsonl",
#                 "w+",
#             ) as outfile:
#                 for split in os.listdir(f"{OUTPUT_DIR}/{dataset_name}/{language}/rephrased"):
#                     print(f"Merging {split}")
#                     with open(
#                         f"{OUTPUT_DIR}/{dataset_name}/{language}/rephrased/{split}", "r"
#                     ) as infile:
#                         for line in infile:
#                             data = json.loads(line)
#                             outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
#                             count += 1

#             print(f"Total samples merged and saved: {count}")

#             print(f"All files merged into {OUTPUT_DIR}/{dataset_name}/{language}/train.jsonl")
            
#             shutil.rmtree(f"{OUTPUT_DIR}/{dataset_name}/{language}/rephrased")

#             shutil.rmtree(f"{OUTPUT_DIR}/{dataset_name}/{language}/raw_data/")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="run command r repharse on cauldron datasets"
#     )

#     parser.add_argument(
#         "--max_tokens",
#         type=int,
#         default=1024,
#         help="Maximum number of tokens to generate",
#     )
#     parser.add_argument(
#         "--temperature",
#         type=float,
#         default=0.5,
#         help="Temperature for sampling",
#     )
#     parser.add_argument(
#         "--top_p",
#         type=float,
#         default=0.9,
#         help="Top p for sampling",
#     )
#     parser.add_argument(
#         "--num_threads",
#         type=int,
#         default=24,
#         help="Number of threads to use for preprocessing",
#     )
#     args = parser.parse_args()

#     print(args)
#     main(
#         args.max_tokens,
#         args.temperature,
#         args.top_p,
#         args.num_threads,
    # )

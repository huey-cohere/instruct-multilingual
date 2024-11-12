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
        raw_cell,
        translated_cell,
        language,
    ):

        message = PROMPT.format(raw_text=raw_cell, translation=translated_cell, language=language)
        
        # print(message)

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
                rephrase_output = response.text.strip()
                match_rephrase_output = re.search(r'Rephrased Translation[:：∶﹕]([\s\S]*)', rephrase_output)
                if match_rephrase_output:
                    rephrase_output_extract = match_rephrase_output.group(1).strip()
                    return rephrase_output_extract
                else:
                    raise Exception("No match found")
            except Exception as e:
                print(f"API Error: {e}")
                print(f"Retry count: {num_retries}")
                print("Retrying in 3 seconds")
                if num_retries == 28:
                    print(f"Failed: {response}")
                await asyncio.sleep(3)
                num_retries += 1

        return None
                
    async def _get_completion_async(
        self,
        translated_tables,
        raw_tables,
        language,
    ):
        
        rephrased_tables = []
        for index, _ in enumerate(translated_tables):
            
            rephrased_table = []

            translated_table = translated_tables[index]
            raw_table = raw_tables[index]

            for i, _ in enumerate(translated_table):
                rephrased_row = []
                for j, _ in enumerate(translated_table[i]):
                    translated_cell = translated_table[i][j]
                    raw_cell = raw_table[i][j]
                    if bool(re.fullmatch(r'^[\d\W_]*$', str(raw_cell))):
                        rephrased_row.append(raw_cell)
                    else:
                        rephrased_cell = await self._get_completion_from_client(
                            raw_cell=raw_cell,
                            translated_cell = translated_cell,
                            language = language,
                        )
                        if rephrased_cell is not None:
                            rephrased_row.append(rephrased_cell)
                        else:
                            rephrased_row.append(translated_cell)
                
                rephrased_table.append(rephrased_row)
                
            rephrased_tables.append(rephrased_table)

        return rephrased_tables


    def get_batch_completions(
        self,
        translated_tables_all,
        raw_tables_all,
        language,
    ):

        async def process_batch():
            semaphore = asyncio.Semaphore(self.max_workers)

            async def bounded_get_completion(translated_tables, raw_tables, index):
                async with semaphore:
                    result = await self._get_completion_async(
                        translated_tables= translated_tables,
                        raw_tables = raw_tables,
                        language = language,
                    )
                    return index, result

            tasks = [bounded_get_completion(translated_tables, raw_tables, i) for i, (translated_tables, raw_tables)  in enumerate(zip(translated_tables_all, raw_tables_all))]

            completions = [None] * len(raw_tables_all)
            for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                index, result = await task
                completions[index] = result
            return completions

        return self.run_on_event_loop(process_batch())
    
    # def translate_tables(self, translated_tables_all, raw_tables_all, language):
    #     rephrased_tables_all = []
    #     for i, (translated_tables, raw_tables) in enumerate(zip(translated_tables_all, raw_tables_all)):
    #         rephrased_tables = self.get_batch_completions(
    #             translated_tables = translated_tables,
    #             raw_tables = raw_tables,
    #             language = language,
    #         )
    #         rephrased_tables_all.append(rephrased_tables)

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
        max_completion_tokens=1024, 
        engine='command-r-plus', 
        base_url="https://stg.api.cohere.com",
        max_workers = 64,
        )

    for lang in os.listdir(args.dataset_path):
        if lang not in ['heb_Hebr', 'ita_Latn', 'ron_Latn']:
            continue

        print(f"Rephrasing {lang}")

        language = Language.get(lang).display_name()

        df = pd.read_json(os.path.join(args.dataset_path, lang, "train.jsonl"), lines=True)
    
        completions = provider.get_batch_completions(
            translated_tables_all = df['Translated_Table'].tolist(),
            raw_tables_all = df['Table'].tolist(),
            language = language,
        )

        # completions = [c[0] for c in completions]

        df = pd.read_json(os.path.join(args.dataset_path, lang, "train.jsonl"), lines=True)

        df['Translated_Table'] = completions

        with open(os.path.join(args.output_dir, lang, "train.jsonl"), 'w', encoding='utf-8') as file:
            df.to_json(file, orient="records", lines=True,  force_ascii=False)


    # language = Language.get('zho_Hant').display_name()

    # # df = pd.read_jsonl(os.path.join(args.dataset_path, lang, "train.jsonl"))
    # df = pd.read_json('/home/olivernan_cohere_com/train.jsonl', lines=True)
    # df = df.head(5)
    # completions = provider.get_batch_completions(
    #     translated_tables_all = df['Translated_Table'].tolist(),
    #     raw_tables_all = df['Table'].tolist(),
    #     language = language,
    # )
    # # print(df['Translated_Table'].tolist())
    # # print("===")
    # # print(completions)
    

    # df = pd.read_json(os.path.join(args.dataset_path, lang, "train.jsonl"), lines=True)

    # df['Translated_Table'] = completions

    # with open(args.output_dir, 'w', encoding='utf-8') as file:
    #     df.to_json(file, orient="records", lines=True,  force_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate datasets from huggingface hub using a variety of methods.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset file", default='/home/olivernan_cohere_com/recap_data_translation_2024_11_01_raw_repharsed_table_translated/RecapMultiHiertt_translation')
    parser.add_argument("--output_dir", type=str, default='/home/olivernan_cohere_com/recap_data_translation_2024_11_01_raw_repharsed_table_translated_rephrase/RecapMultiHiertt_translation', help="Output directory for the translated dataset")
    args = parser.parse_args()
    print(args)
    main(args)










# import os
# import random
# import time
# from collections import defaultdict
# from typing import Dict, List, Set
# import requests
# import json
# from tqdm import tqdm
# import argparse
# import logging
# from cohere import AsyncClient
# import re
# import pandas as pd
# import base64
# import io
# from PIL import Image
# import asyncio

# logging.basicConfig(
#     level=logging.ERROR,
#     format="%(asctime)s - %(levelname)s - %(message)s",
# )


# client = AsyncClient('XvksMYmZbQF0aagROSgmbTbiL4auprJmQtWk7iFP', base_url="https://stg.api.cohere.com") # "https://stg.api.cohere.ai"

# PROMPT =  """Original Text: 
# {raw_text}\n

# Translation: 
# {translation}\n

# Instruction:
# Given the original text and its translation, improve the quality of the translation by rephrasing it. 
# Ensure the rephrased translation closely aligns with the original text in meaning, structure, tone, and style. 
# Make the rephrased translation sound natural and fluent in the target language while preserving all essential details, correcting any grammatical errors, and retaining all stylistic elements (e.g., enumeration, parentheses, punctuation, capitalization, spacing, line breaks, etc.) from the original.

# The output must strictly follow this format:
# Rephrased Translation: <rephrased translation placeholder>"""



# async def rephrase(raw_text, translation, engine, temperature, top_p, max_tokens):

#     rephrase_input = PROMPT.format(raw_text=raw_text, translation=translation)

#     retry_count = 0
#     response_output = None
#     while retry_count < 30:
#         try:
#             response_output = await client.chat(
#                 model=engine,
#                 messages=[
#                     {
#                         "role": "user",
#                         "content": rephrase_input
#                     }
#                 ],
#                 temperature = temperature,
#                 p = top_p,
#                 max_tokens = max_tokens,
#             )
#             rephrase_output = response_output.message.content[0].text.strip()
#             match_rephrase_output = re.search(r'Rephrased Translation[:：∶﹕]([\s\S]*)', rephrase_output)
#             if match_rephrase_output:
#                 rephrase_output_extract = match_rephrase_output.group(1).strip()
#             else:
#                 raise Exception("No match found")
            
#             return rephrase_output_extract
        
#         except Exception as e:
#             logging.error(f"API Error: {e}")
#             logging.error(f"Retry count: {retry_count}")
#             logging.error("Retrying in 3 seconds")
#             if retry_count == 28:
#                 logging.error(f"Failed: {response_output}")
#                 logging.error(f"raw_text: {raw_text}")
#                 logging.error(f"nllb translation: {translation}")
#             time.sleep(3)
#             retry_count += 1
    
#     return None

# def translate_dataset_via_api(
#     inputs
# ) -> Dict[str, List[str]]:

#     translated_table, table, engine, max_tokens, temperature, top_p = inputs

#     rephrased_table = []
#     for i, row in enumerate(table):
#         translate_row = []
#         for j, _ in enumerate(row):
#             translated_cell = translated_table[i][j]
#             raw_cell = table[i][j]
#             if bool(re.fullmatch(r'^[\d\W_]*$', str(cell))):
#                 translate_row.append(cell)
#             else:
#                 rephrased_cell = rephrase(raw_cell, translated_cell, engine, temperature, top_p, max_tokens)

#                 if rephrased_cell is not None:
#                     translate_row.append(rephrased_cell)
#                 else:
#                     translate_row.append(translated_cell)

#         translated_table.append(translate_row)

#     return translated_table

# def translate_dataset_via_inference_api(
#     dataset_path,
#     max_tokens: int,
#     temperature: float,
#     top_p: float,
#     engine: str,
#     output_dir: str = "./datasets",
# ) -> None:

#     start_time = time.time()

#     size = 0
#     with open(dataset_path, "r") as file, open(output_dir, "w") as f:
#         for i, line in enumerate(tqdm(file, desc="translating", unit="sample")):
#             translated_table_images = []
#             translated_tables = []
#             data = json.loads(line)
#             tables = data['Table']
#             for index, table in enumerate(tables):
#                 translated_table = translate_table((table, url, engine, source_language_code, target_language_code, max_tokens, temperature, top_p))
#                 translated_tables.append(translated_table)
#             data['Translated_Table'] = translated_tables
#             f.write(json.dumps(data, ensure_ascii=False) + "\n")
#             size += 1

#     print(f"Translated {size} samples")

#     end_time = time.time()
#     elapsed_time = end_time - start_time
    
#     print(f"Elapsed time: {elapsed_time:.4f} seconds")
#     return translate_table


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Translate datasets from huggingface hub using a variety of methods.")
#     parser.add_argument("--dataset_path", type=str, help="Path to the dataset file")
#     parser.add_argument("--output_dir", type=str, default="./datasets", help="Output directory for the translated dataset")
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
#     args = parser.parse_args()
#     print(args)

#     translate_dataset_via_api(
#         dataset_path=args.dataset_path,
#         output_dir=args.output_dir,
#         max_tokens=args.max_tokens,
#         temperature=args.temperature,
#         top_p=args.top_p,
#         engine='command-r-plus',
#     )


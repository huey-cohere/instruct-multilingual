
import argparse
from tqdm import tqdm
import os
import time
import concurrent.futures
import json
import io
import cohere
import re
import functools
import gcsfs
import random
import logging
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
@functools.lru_cache(maxsize=1)
def gcfs():
    return gcsfs.GCSFileSystem()

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

def make_request(params):

    data_dict, engine, target_language_code, max_tokens, temperature, top_p = params

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

    formatted_input_user = PROMPT.format(raw_text=raw_user, translation=translated_user)
    formatted_input_chatbot = PROMPT.format(raw_text=raw_chatbot, translation=translated_chatbot)

    retry_count = 0
    response_user = None
    response_chatbot = None
    while retry_count < 30:
        try:
            response_user = client.chat(
                model=engine,
                messages=[
                    {
                        "role": "user",
                        "content": formatted_input_user
                    }
                ],
                temperature = temperature,
                p = top_p,
                max_tokens = max_tokens,
            )
            output_user = response_user.message.content[0].text.strip()
            match_user = re.search(r'Rephrased Translation[:：∶﹕]([\s\S]*)', output_user)
            if match_user:
                response_user_extract = match_user.group(1).strip()
            else:
                raise Exception("No match found")
            
            data_dict['User'].append(
                {
                    "text": response_user_extract,
                    "language": target_language_code,
                    "source": "raw-processed-nllb_translated-command_r_rephrase",
                }
            )
            
            response_chatbot = client.chat(
                model=engine,
                messages=[
                    {
                        "role": "user",
                        "content": formatted_input_chatbot
                    }
                ],
                temperature = temperature,
                p = top_p,
                max_tokens = max_tokens,
            )
            output_chatbot = response_chatbot.message.content[0].text.strip()
            match_chatbot = re.search(r'Rephrased Translation[:：∶﹕]([\s\S]*)', output_chatbot)
            if match_chatbot:
                response_chatbot_extract = match_chatbot.group(1).strip()
            else:
                raise Exception("No match found")

            data_dict['Chatbot'].append(
                {
                    "text": response_chatbot_extract,
                    "language": target_language_code,
                    "source": "raw-gpt_recap-nllb_translated-command_r_rephrase",
                }
            )

            return data_dict
        
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


def run_query(
    num_threads,
    output_dir,
    dataset_path,
    engine,
    max_tokens,
    temperature,
    top_p,
    target_language_code,
):
    # with gcfs().open(dataset_path, "r") as file:
    with open(dataset_path, "r") as file:
        dataset = [json.loads(line) for line in file]

    # dataset = dataset[:10]
    print(f"dataset size: {len(dataset)}")
    # print(dataset[0])
    # print(dataset[1])

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:

        futures = [
            executor.submit(
                make_request, (data, engine, target_language_code, max_tokens, temperature, top_p)
            )
            for data in dataset
        ]

        new_dataset = []
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            result = future.result()
            if result is not None:
                new_dataset.append(result)

    print(new_dataset[0])
    print(f"new dataset size: {len(new_dataset)}")

    with open(output_dir, "w+") as f:
        for data in new_dataset:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run generation on a dataset")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to the dataset",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=48,
        help="Number of threads to use for preprocessing",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/olivernan/",
        help="Output directory for generated data",
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
        default=0.8,
        help="Top p for sampling",
    )
    parser.add_argument(
        "--target_language_code",
        type=str
    )

    args = parser.parse_args()
    print(args)

    # run_inference(num_proc=args.num_proc, output_dir=args.output_dir, dataset_path=args.dataset_path)

    run_query(
        num_threads=args.num_threads,
        output_dir=args.output_dir,
        dataset_path=args.dataset_path,
        engine='command-r-plus',
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        target_language_code=args.target_language_code,
    )

    # python gpt_recaption.py --dataset_path /home/olivernan_cohere_com/gpt-recaption/CauldronAi2d/raw_data/train.jsonl --num_threads 64 --output_dir /home/olivernan_cohere_com/gpt-recaption/CauldronAi2d/2024_09_09/train.jsonl --max_tokens 512 --temperature 1.0 --top_p 0.95 --n 1


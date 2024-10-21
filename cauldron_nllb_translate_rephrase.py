
import argparse
from tqdm import tqdm
import os
import time
import concurrent.futures
import json
import io
import cohere
import re

client = cohere.ClientV2("EjAoSdeyYwowsjVbf2YywU5dZZhXX1RYi5umpN5x", base_url="https://stg.api.cohere.ai")


PROMPT = """
Original Sentence: {raw_sentene}\n
Translated Sentence: {translated_sentene}\n

Rephrase the translated sentence to enhance its quality, ensuring it aligns closely with the original in meaning, structure, tone, and style.
Ensure that the rephrased sentence conveys the same meaning as the original sentence but avoid altering the core message or introducing new information. 
Correct any grammatical errors present in the translated sentence.
Maintain a structure similar to the original sentence. 
Match the tone and style of the original sentence. 
Preserve any stylistic elements such as enumeration, punctuation or capitalization.

The output must strictly follow this format:\n
Rephrase Translated Sentence: <sentence>"""


def make_request(params):

    data_dict, engine, target_language_code, max_tokens, temperature, top_p = params

    for user in data_dict['User']:
        if user['language'] == "eng_Latn" and user['source'] == "raw-processed":
            eng_Latn_user = user['text']
        if user['language'] == target_language_code and user['source'] == "raw-processed-nllb_translated":
            translated_user = user['text']
    
    for bot in data_dict['Chatbot']:
        if bot['language'] == "eng_Latn" and bot['source'] == "raw-gpt_recap":
            eng_Latn_chatbot = bot['text']
        if bot['language'] == target_language_code and bot['source'] == "raw-gpt_recap-nllb_translated":
            translated_chatbot = bot['text']

    formatted_input_user = PROMPT.format(raw_sentene=eng_Latn_user, translated_sentene=translated_user)
    formatted_input_chatbot = PROMPT.format(raw_sentene=eng_Latn_chatbot, translated_sentene=translated_chatbot)

    retry_count = 0
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
            output_user = response_user.choices[0].message.content.strip()
            match_user = re.search(r'Rephrase Translated Sentence:\s*(.+)', output_user)
            if match_user:
                response_user_extract = match_user.group(1)
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
            output_chatbot = response_chatbot.choices[0].message.content.strip()
            match_chatbot = re.search(r'Rephrase Translated Sentence:\s*(.+)', output_chatbot)
            if match_chatbot:
                response_chatbot_extract = match_chatbot.group(1)
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
            print(f"API Error: {e}")
            print(f"count: {retry_count}")
            print(f"Retring in 10 seconds")
            time.sleep(10)
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

    with open(dataset_path, "r") as file:
        dataset = [json.loads(line) for line in file]

    # dataset = dataset[:10]
    print(f"dataset size: {len(dataset)}")
    print(dataset[0])
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
            f.write(json.dumps(data) + "\n")


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
        default="/home/olivernan/multimodal_generated_data",
        help="Output directory for generated data",
    )
    parser.add_argument(
        "--engine",
        type=str,
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
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
        engine=args.engine,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        target_language_code=args.target_language_code,
    )

    # python gpt_recaption.py --dataset_path /home/olivernan_cohere_com/gpt-recaption/CauldronAi2d/raw_data/train.jsonl --num_threads 64 --output_dir /home/olivernan_cohere_com/gpt-recaption/CauldronAi2d/2024_09_09/train.jsonl --max_tokens 512 --temperature 1.0 --top_p 0.95 --n 1


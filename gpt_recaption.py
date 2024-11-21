import datasets
import argparse
from tqdm import tqdm
from openai import OpenAI
import multiprocessing
import functools
import os
import openai
import time
import concurrent.futures
import json
import io
import abc
import asyncio
from dataclasses import dataclass
from typing import Optional, Union
import re
import pandas as pd
from langcodes import Language



os.environ[
    "OPENAI_API_KEY"
] = "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"

client = OpenAI()


def make_request(params):

    data, engine, max_tokens, temperature, top_p, n = params

    images = []
    question = None
    answer = None

    # for turn in data["turns"]:
    #     if turn["role"] == "User":
    #         for content in turn["content"]:
    #             if "text" in content:
    #                 question = content["text"]
    #             if "url" in content:
    #                 images.append(content["url"])
    #     if turn["role"] == "Chatbot":
    #         answer = turn["content"]

    for user in data['User']:
        if user['language'] == "eng_Latn" and user['source'] == "raw-processed":
            question = user['text']
            break
    
    for chatbot in data['Chatbot']:
        if chatbot['language'] == "eng_Latn" and chatbot['source'] == "raw":
            answer = chatbot['text']
            break
    
    for image in data['Image']:
        images.append(image)
            
    assert question is not None
    assert answer is not None
    assert len(images) > 0

    formatted_input = f"Instruction: {question.strip()}.\nReference Answer: {answer.strip()}.\nTask: Given the instruction and reference answer, generate a more detailed, informative, and well-explained response. Ensure the response is clear, insightful, and expands upon the reference answer with relevant additional information."

    query = {
        "role": "user",
        "content": [
            {"type": "text", "text": formatted_input},
        ],
    }

    for url in images:
        image_dict = {
            "type": "image_url",
            "image_url": {
                "url": url,
            },
        }
        query["content"].append(image_dict)

    retry_count = 0

    # output = {"User": [], "Chatbot": []}
    # output["Image"] = images
    # output["User"].append({"text": question, "language": "eng_Latn", "source": "raw"})
    # output["Chatbot"].append({"text": answer, "language": "eng_Latn", "source": "raw"})

    while retry_count < 30:
        try:
            response = client.chat.completions.create(
                model=engine,
                messages=[query],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                n=n,
            )
            # output = {
            #     "role": "Chatbot-gpt",
            #     "content": response.choices[0].message.content.strip(),
            # }
            # input_data["turns"].append(output)

            # cost = (
            #     response.usage.prompt_tokens * PRICE_PER_INPUT_TOKEN
            #     + response.usage.completion_tokens * PRICE_PER_OUTPUT_TOKEN
            # )

            # return (input_data, cost)

            # output["Chatbot"].append(
            #     {
            #         "text": response.choices[0].message.content.strip(),
            #         "language": "eng_Latn",
            #         "source": "raw-gpt_recap",
            #     }
            # )
            
            data["Chatbot"].append(
                {
                    "text": response.choices[0].message.content.strip(),
                    "language": "eng_Latn",
                    "source": "raw-gpt_recap",
                }
            )

            cost = (
                response.usage.prompt_tokens * PRICE_PER_INPUT_TOKEN
                + response.usage.completion_tokens * PRICE_PER_OUTPUT_TOKEN
            )

            # return (output, cost)
            return (data, cost)
        except (
            openai.APIError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.InternalServerError,
        ) as e:
            print(f"OpenAI API returned an API Error: {e}")
            print(f"count: {retry_count}")
            print(f"Retring in 30 seconds")
            time.sleep(30)
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
    n,
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
                make_request, (data, engine, max_tokens, temperature, top_p, n)
            )
            for data in dataset
        ]

        new_dataset = []
        cost_total = []
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            result = future.result()
            if result is not None:
                generation, cost = result
                new_dataset.append(generation)
                cost_total.append(cost)

    print(new_dataset[0])
    print(f"new dataset size: {len(new_dataset)}")
    print(f"cost: {sum(cost_total)}")
    print(f"cost per query: {sum(cost_total) / len(new_dataset)}")

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
        default=1024,
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
        "--n",
        type=int,
        default=1,
        help="Number of completions to generate",
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
        n=args.n,
    )

    # python gpt_recaption.py --dataset_path /home/olivernan_cohere_com/gpt-recaption/CauldronAi2d/raw_data/train.jsonl --num_threads 64 --output_dir /home/olivernan_cohere_com/gpt-recaption/CauldronAi2d/2024_09_09/train.jsonl --max_tokens 512 --temperature 1.0 --top_p 0.95 --n 1

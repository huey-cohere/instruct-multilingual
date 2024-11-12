import yaml
import ast
import os
from datetime import date
import argparse
from google.cloud import storage
import subprocess
import json
import shutil
import random
import math


DATA_DIR = "/home/olivernan_cohere_com/recap_data_translation_2024_11_01_raw_rephrased"

OUTPUT_DIR = "/home/olivernan_cohere_com/recap_data_translation_2024_11_01_raw_repharsed_table_translated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# os.makedirs("finqa_translated_images", exist_ok=True)
os.makedirs("logs/", exist_ok=True)

SERVER_PORT_LIST = [f"http://localhost:{8000 + i}/translate" for i in range(64)]

def main():

    for dataset_name in os.listdir(DATA_DIR):
        
        if dataset_name not in ['RecapMultiHiertt_translation']:
            continue

        for lang in os.listdir(f"{DATA_DIR}/{dataset_name}"):

            if lang not in ['heb_Hebr', 'ita_Latn', 'ron_Latn']:
                continue

            print(f"Dataset: {dataset_name} Language: {lang}")

            with open(os.path.join(DATA_DIR, dataset_name, lang, 'train.jsonl'), "r") as file:
                dataset = [json.loads(line) for line in file]
        
            print(f"Dataset size: {len(dataset)}")
            
            num_per_server = math.ceil(len(dataset) / (len(SERVER_PORT_LIST)))
            dataset_chunk = [
                dataset[i : i + num_per_server] for i in range(0, len(dataset), num_per_server)
            ]
            dataset_paths = []
            os.makedirs(f"{OUTPUT_DIR}/{dataset_name}/{lang}/raw_data", exist_ok=True)
            for i, chunk in enumerate(dataset_chunk):
                _path = f"{OUTPUT_DIR}/{dataset_name}/{lang}/raw_data/split_{i}.jsonl"
                dataset_paths.append(_path)
                with open(_path, "w+") as f:
                    for i, line in enumerate(chunk):
                        f.write(json.dumps(line, ensure_ascii=False) + "\n")

            del dataset_chunk
            del dataset

            os.makedirs(f"{OUTPUT_DIR}/{dataset_name}", exist_ok=True)

            processes = []
            # for i, path in enumerate(dataset_paths):
            for i, path in enumerate(dataset_paths):
                
                port_url = SERVER_PORT_LIST[i % len(SERVER_PORT_LIST)]

                output_dir = (
                    f"{OUTPUT_DIR}/{dataset_name}/{lang}/translated_splits/split_{i}.jsonl"
                )
                
                os.makedirs(f"{OUTPUT_DIR}/{dataset_name}/{lang}/translated_splits/", exist_ok=True)

                command = f"""
                nohup python table_translate.py \
                --dataset_path {path} \
                --target_language_code {lang} \
                --source_language_code eng_Latn \
                --url {port_url} \
                --output_dir {output_dir} > logs/table-{dataset_name}_{lang}_{i}.out
                """

                print("Running generation command")
                print(command)
                process = subprocess.Popen(command, shell=True)

                processes.append(process)

                if len(processes) == (len(SERVER_PORT_LIST)):
                    for process in processes:
                        process.wait()
                    processes = [] 
            

            for process in processes:
                process.wait()

            count = 0
            with open(
                f"{OUTPUT_DIR}/{dataset_name}/{lang}/train.jsonl",
                "w+",
            ) as outfile:
                for split in os.listdir(f"{OUTPUT_DIR}/{dataset_name}/{lang}/translated_splits"):
                    print(f"Merging {split}")
                    with open(
                        f"{OUTPUT_DIR}/{dataset_name}/{lang}/translated_splits/{split}", "r"
                    ) as infile:
                        for line in infile:
                            data = json.loads(line)
                            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
                            count += 1

            print(f"Total samples merged and saved: {count}")

            print(f"All files merged into {OUTPUT_DIR}/{dataset_name}/{lang}/train.jsonl")
            
            shutil.rmtree(f"{OUTPUT_DIR}/{dataset_name}/{lang}/translated_splits/")

            shutil.rmtree(f"{OUTPUT_DIR}/{dataset_name}/{lang}/raw_data/")


        # break


if __name__ == "__main__":

    # parser = argparse.ArgumentParser(
    #     description="run command r repharse on cauldron datasets"
    # )

    # parser.add_argument(
    #     "--max_tokens",
    #     type=int,
    #     default=1024,
    #     help="Maximum number of tokens to generate",
    # )
    # parser.add_argument(
    #     "--temperature",
    #     type=float,
    #     default=0.5,
    #     help="Temperature for sampling",
    # )
    # parser.add_argument(
    #     "--top_p",
    #     type=float,
    #     default=0.9,
    #     help="Top p for sampling",
    # )
    # args = parser.parse_args()

    # print(args)
    main(
        # args.max_tokens,
        # args.temperature,
        # args.top_p,
    )



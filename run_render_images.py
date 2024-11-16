import yaml
import ast
import os
from datetime import date
import argparse
import subprocess
import json
import shutil
import random
import math

FOLDER = "/home/olivernan_cohere_com/recap_data_translation_2024_11_01_raw_repharsed_table_translated/RecapMultiHiertt_translation"
OUTPUT_BASE = "/home/olivernan_cohere_com/recap_data_translation_2024_11_01_raw_repharsed_table_translated_rendered_table/RecapMultiHiertt_translation"
os.makedirs(OUTPUT_BASE, exist_ok=True)

os.makedirs("logs/", exist_ok=True)

language_list =[
    "arb_Arab",
    "zho_Hans",
    "zho_Hant",
    "ces_Latn",
    "nld_Latn",
    "fra_Latn",
    "deu_Latn",
    "ell_Grek",
    "heb_Hebr",
    "hin_Deva",
    "ind_Latn",
    # "ita_Latn",
    # "jpn_Jpan",
    # "kor_Hang",
    # "pes_Arab",
    # "pol_Latn",
    # "por_Latn",
    # "ron_Latn",
    # "rus_Cyrl",
    # "spa_Latn",
    # "tur_Latn",
    # "ukr_Cyrl",
    # "vie_Latn",
]


def main():
    processes = []
    for language in os.listdir(FOLDER):

        if language not in language_list:
            continue

        print(f"Language: {language}")
        os.makedirs(f"{OUTPUT_BASE}/{language}", exist_ok=True)
        
        input_path = os.path.join(FOLDER, language, "train.jsonl") 
        output_path = os.path.join(OUTPUT_BASE, language, "train.jsonl")

        command = f"""
        nohup python render_images.py \
        --input_path {input_path} \
        --output_path {output_path} > logs/{language}.out
        """

        print("Running generation command")
        print(command)
        process = subprocess.Popen(command, shell=True)
        process.wait()
    #     processes.append(process)
    
    # for process in processes:
    #     process.wait()


        # with open(os.path.join(FOLDER, language, "train.jsonl"), "r") as f:
        #     dataset = [json.loads(line) for line in f]

        # print(f"Dataset size: {len(dataset)}")
        
        # num_per_chunk = 128
        # dataset_chunk = [
        #     dataset[i : i + num_per_chunk] for i in range(0, len(dataset), num_per_chunk)
        # ]

        # dataset_paths = []
        # os.makedirs(f"{OUTPUT_BASE}/{language}/raw_splits", exist_ok=True)
        # for i, chunk in enumerate(dataset_chunk):
        #     _path = f"{OUTPUT_BASE}/{language}/raw_splits/{i}.jsonl"
        #     dataset_paths.append(_path)
        #     with open(_path, "w+") as f:
        #         for i, line in enumerate(chunk):
        #             f.write(json.dumps(line, ensure_ascii=False) + "\n")

        # del dataset_chunk
        # del dataset

        # processes = []
        # os.makedirs(f"{OUTPUT_BASE}/{language}/output_splits", exist_ok=True)

        # for i, path in enumerate(dataset_paths):
            
        #     output_path = f"{OUTPUT_BASE}/{language}/output_splits/{i}.jsonl"
        
        #     command = f"""
        #     nohup python render_images.py \
        #     --input_path {path} \
        #     --output_path {output_path} > logs/{language}_{i}.out
        #     """

        #     print("Running generation command")
        #     print(command)
        #     process = subprocess.Popen(command, shell=True)
        #     process.wait()
        #     # processes.append(process)
    

        # # for process in processes:
        # #     process.wait()


        # count = 0
        # with open(
        #     f"{OUTPUT_BASE}/{language}/train.jsonl",
        #     "w+",
        # ) as outfile:
        #     for split in os.listdir(f"{OUTPUT_BASE}/{language}/output_splits"):
        #         print(f"Merging {split}")
        #         with open(
        #             f"{OUTPUT_BASE}/{language}/output_splits/{split}", "r"
        #         ) as infile:
        #             for line in infile:
        #                 data = json.loads(line)
        #                 outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
        #                 count += 1

        # print(f"Total samples merged and saved: {count}")

        # print(f"All files merged into {OUTPUT_BASE}/{language}/train.jsonl")
        
        # shutil.rmtree(f"{OUTPUT_BASE}/{language}/output_splits")

        # shutil.rmtree(f"{OUTPUT_BASE}/{language}/raw_splits")


if __name__ == "__main__":

    main()

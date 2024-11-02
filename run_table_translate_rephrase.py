# from instructmultilingual.translate_cauldron import translate_dataset

# translate_dataset(
#     dataset_path="train.jsonl",
#     url= "http://localhost:8000/translate",
#     output_dir= "/home/olivernan_cohere_com/instruct-multilingual/datasets",
# )

aya23_code2lang ={
    "arb_Arab": "Modern Standard Arabic",
    "zho_Hans": "Chinese (Simplified)",
    "zho_Hant": "Chinese (Traditional)",
    "ces_Latn": "Czech",
    "nld_Latn": "Dutch",
    # "eng_Latn": "English",
    "fra_Latn": "French",
    "deu_Latn": "German",
    "ell_Grek": "Greek",
    "heb_Hebr": "Hebrew",
    "hin_Deva": "Hindi",
    "ind_Latn": "Indonesian",
    "ita_Latn": "Italian",
    "jpn_Jpan": "Japanese",
    "kor_Hang": "Korean",
    "pes_Arab": "Western Persian",
    "pol_Latn": "Polish",
    "por_Latn": "Portuguese",
    "ron_Latn": "Romanian",
    "rus_Cyrl": "Russian",
    "spa_Latn": "Spanish",
    "tur_Latn": "Turkish",
    "ukr_Cyrl": "Ukrainian",
    "vie_Latn": "Vietnamese",
}


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

# CAULDRON_RECAPTION_SOURCE = (
#     "/home/olivernan_cohere_com/instruct-multilingual/source.yaml"
# )

DATA_DIR = "/home/olivernan_cohere_com/recap_data_translation_2024_11_01_raw_rephrased"

OUTPUT_DIR = "/home/olivernan_cohere_com/recap_data_translation_2024_11_01_raw_repharsed_table_translated"

SERVER_PORT_LIST = [f"http://localhost:{8000 + i}/translate" for i in range(64)]

def main():

    for dataset_name in os.listdir(DATA_DIR):

        print(f"Dataset: {dataset_name}")

        for lang in os.listdir(f"{DATA_DIR}/{dataset_name}"):
            print(f"Language: {lang}")

            with open(os.path.join(DATA_DIR, dataset_name, lang, 'train.jsonl'), "r") as file:
                dataset = [json.loads(line) for line in file]
        
            print(f"Dataset size: {len(dataset)}")

            
            num_per_server = math.ceil(len(dataset) / len(SERVER_PORT_LIST))
            dataset_chunk = [
                dataset[i : i + num_per_server] for i in range(0, len(dataset), num_per_server)
            ]
            dataset_paths = []
            os.makedirs(f"{OUTPUT_DIR}/{dataset_name}/{lang}/raw_data/splits", exist_ok=True)
            for i, chunk in enumerate(dataset_chunk):
                _path = f"{DATA_DIR}/{dataset_name}_translation/raw_data/splits/split_{i}.jsonl"
                dataset_paths.append(_path)
                with open(_path, "w+") as f:
                    for i, line in enumerate(chunk):
                        f.write(json.dumps(line, ensure_ascii=False) + "\n")

            del dataset_chunk
            del dataset
            
            os.makedirs(f"{DATA_DIR}/{dataset_name}_translation", exist_ok=True)
            os.makedirs("logs/", exist_ok=True)
            # dataset_paths = [f"{DATA_DIR}/{dataset_name}_translation/raw_data/splits/{path}" for path in os.listdir(f"{DATA_DIR}/{dataset_name}/raw_data/splits")]
            for code, language in aya23_code2lang.items():
                # if language != "English":
                print(f"Currently translating: {language}")
                processes = []
                # for i, path in enumerate(dataset_paths):
                for i, path in enumerate(dataset_paths[code]):
                    
                    # port_url = SERVER_PORT_LIST[(i // 2) % len(SERVER_PORT_LIST)]
                    port_url = SERVER_PORT_LIST[i % len(SERVER_PORT_LIST)]

                    output_dir = (
                        f"{DATA_DIR}/{dataset_name}_translation/{code}/splits/split_{i}.jsonl"
                    )
                    
                    os.makedirs(f"{DATA_DIR}/{dataset_name}_translation/{code}/splits/", exist_ok=True)

                    command = f"""
                    nohup python instructmultilingual/translate_cauldron.py \
                    --dataset_path {path} \
                    --target_language_code {code} \
                    --source_language_code eng_Latn \
                    --url {port_url} \
                    --output_dir {output_dir} > logs/{dataset_name}_{code}_{i}.out
                    """

                    print("Running generation command")
                    print(command)
                    process = subprocess.Popen(command, shell=True)

                    processes.append(process)

                    if len(processes) == len(SERVER_PORT_LIST):
                        for process in processes:
                            process.wait()
                        processes = [] 
                

                for process in processes:
                    process.wait()

                count = 0
                with open(
                    f"{DATA_DIR}/{dataset_name}_translation/{code}/train.jsonl",
                    "w+",
                ) as outfile:
                    for split in os.listdir(f"{DATA_DIR}/{dataset_name}_translation/{code}/splits"):
                        print(f"Merging {split}")
                        with open(
                            f"{DATA_DIR}/{dataset_name}_translation/{code}/splits/{split}", "r"
                        ) as infile:
                            for line in infile:
                                data = json.loads(line)
                                outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
                                count += 1

                print(f"Total samples merged and saved: {count}")

                print(f"All files merged into {DATA_DIR}/{dataset_name}_translation/{code}/train.jsonl")
                
                shutil.rmtree(f"{DATA_DIR}/{dataset_name}_translation/{code}/splits/")

            shutil.rmtree(f"{DATA_DIR}/{dataset_name}_translation/raw_data/")
            # os.rename(f"{DATA_DIR}/{dataset_name}_translation/raw_data/", f"{DATA_DIR}/{dataset_name}_translation/eng_Latn/")
            # shutil.move(f"{DATA_DIR}/{dataset_name}/eng_Latn/", f"{DATA_DIR}/{dataset_name}/translation/")

            # break


if __name__ == "__main__":

    main()



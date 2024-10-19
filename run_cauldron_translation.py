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

CAULDRON_RECAPTION_SOURCE = (
    "/home/olivernan_cohere_com/instruct-multilingual/cauldron_recaption_source.yaml"
)

DATA_DIR = "/home/olivernan_cohere_com/recap_cauldron_translation_2024_10_15_raw"

SERVER_PORT_LIST = [f"http://localhost:{8000 + i}/translate" for i in range(64)]

def main():
    with open(CAULDRON_RECAPTION_SOURCE, "r") as file:
        cauldron_source = yaml.safe_load(file)

    for dataset_name, dataset_info in cauldron_source.items():

        print(f"Dataset: {dataset_name}")

        gs_path = dataset_info["path"]
        print(f"Path: {gs_path}")

        download_save_path = f"{DATA_DIR}/{dataset_name}_translation/raw_data/"
        os.makedirs(download_save_path, exist_ok=True)

        # Download the dataset
        print("Downloading dataset")
        download_command = f"gsutil -m cp -r {gs_path} {download_save_path}"
        print(download_command)
        subprocess.run(
            download_command,
            check=True,
            shell=True,
        )

        with open(os.path.join(download_save_path, 'train.jsonl'), "r") as file:
            dataset = [json.loads(line) for line in file]
        
        downsample_num = dataset_info["downsample_num"] if "downsample_num" in dataset_info else None
        
        # with open(os.path.join('/home/olivernan_cohere_com/recap_cauldron_translation_2024_10_11_raw/RecapCauldronLocalized_narratives_translation/eng_Latn', 'train.jsonl'), "r") as file:
        #     dataset = [json.loads(line) for line in file]
        
        print(f"Dataset size: {len(dataset)}")

        # if downsample_num is not None:
        #     print(f"Downsampling to {downsample_num}")
        #     dataset = random.sample(dataset, downsample_num)
        
        # os.makedirs(f"{DATA_DIR}/{dataset_name}_translation/eng_Latn", exist_ok=True)
        # with open(f"{DATA_DIR}/{dataset_name}_translation/eng_Latn/train.jsonl", "w+") as f:
        #     for i, line in enumerate(dataset):
        #         # example = {"User":[], "Chatbot":[], "Image":[] }
        #         # for turn in line["turns"]:
        #         #     if turn["role"] == "User":
        #         #         for content in turn['content']:
        #         #             if "text" in content:
        #         #                 example['User'].append({"text": content["text"], "language": "eng_Latn", "source": "raw"})
        #         #             if "url" in content:
        #         #                 example['Image'].append(content["url"])
        #         #     if turn["role"] == "Chatbot":
        #         #         example['Chatbot'].append({'text': turn["content"], "language": "eng_Latn", "source": "raw"})
        #         #     if turn["role"] == "Chatbot-gpt":
        #         #         example['Chatbot'].append({"text": turn["content"], "language": "eng_Latn", "source": "raw-gpt_recap"})
                
        #         # dataset[i] = example
        #         # f.write(json.dumps(example) + "\n")

        #         f.write(json.dumps(line) + "\n")   
        #

        
        dataset_paths = {}
        if downsample_num is not None:
            for code, language in aya23_code2lang.items():
                print(f"Downsampling to {downsample_num} for {language}")
                dataset_downsample = random.sample(dataset, downsample_num)
                num_per_server = math.ceil(downsample_num / len(SERVER_PORT_LIST))
                dataset_chunk = [
                    dataset_downsample[i : i + num_per_server] for i in range(0, len(dataset_downsample), num_per_server)
                ]
                os.makedirs(f"{DATA_DIR}/{dataset_name}_translation/raw_data/{code}", exist_ok=True)
                dataset_paths[code] = []
                for i, chunk in enumerate(dataset_chunk):
                    _path = f"{DATA_DIR}/{dataset_name}_translation/raw_data/{code}/split_{i}.jsonl"
                    dataset_paths[code].append(_path)
                    with open(_path, "w+") as f:
                        for i, line in enumerate(chunk):
                            f.write(json.dumps(line) + "\n")
        else:
            num_per_server = math.ceil(len(dataset) / len(SERVER_PORT_LIST))
            dataset_chunk = [
                dataset[i : i + num_per_server] for i in range(0, len(dataset), num_per_server)
            ]
            path_list = []
            os.makedirs(f"{DATA_DIR}/{dataset_name}_translation/raw_data/splits", exist_ok=True)
            for i, chunk in enumerate(dataset_chunk):
                _path = f"{DATA_DIR}/{dataset_name}_translation/raw_data/splits/split_{i}.jsonl"
                path_list.append(_path)
                with open(_path, "w+") as f:
                    for i, line in enumerate(chunk):
                        f.write(json.dumps(line) + "\n")
            for code, language in aya23_code2lang.items():
                dataset_paths[code] = path_list

                
        
        # print(f"Dataset size: {len(dataset)}")

        # dataset_chunk = [
        #     dataset[i : i + num_per_server] for i in range(0, len(dataset), num_per_server)
        # ]
        # dataset_chunk = []

        # dataset_paths = []
        # for i, chunk in enumerate(dataset_chunk):
        #     os.makedirs(f"{DATA_DIR}/{dataset_name}_translation/raw_data/splits", exist_ok=True)
        #     _path = f"{DATA_DIR}/{dataset_name}_translation/raw_data/splits/split_{i}.jsonl"
        #     dataset_paths.append(_path)
        #     with open(_path, "w+") as f:
        #         for line in chunk:
        #             f.write(json.dumps(line) + "\n")            

        # dataset_paths = []
        # for i, chunk in enumerate(dataset_chunk):
        #     os.makedirs(f"{DATA_DIR}/{dataset_name}_translation/raw_data/splits", exist_ok=True)
        #     _path = f"{DATA_DIR}/{dataset_name}_translation/raw_data/splits/split_{i}.jsonl"
        #     dataset_paths.append(_path)
        #     with open(_path, "w+") as f:
        #         for line in chunk:
        #             f.write(json.dumps(line) + "\n")

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



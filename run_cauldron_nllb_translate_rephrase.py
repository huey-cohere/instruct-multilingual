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
                    'RecapFINAQA_translation',
                    'RecapMultiHiertt_translation',
                    ]

def main(
    max_tokens: int,
    temperature: float,
    top_p: float,
    num_threads: int,
):
    os.makedirs("logs/", exist_ok=True)

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
                --top_p {top_p} > logs/{dataset_name}_{language}.out
                """

            print("Running command")
            print(command)
            process = subprocess.Popen(command, shell=True)
            process.wait()

        #     processes.append(process)

        #     if len(processes) == 2:
        #         for process in processes:
        #             process.wait()
        #         processes = [] 
        
        # for process in processes:
        #     process.wait()

        #     break
        # break

            # download_save_path = f"{DATA_DIR}/{dataset_name}/raw_data/train.jsonl"

            # # Download the dataset
            # print("Downloading dataset")
            # download_command = f"gsutil -m cp -r {gs_path} {download_save_path}"
            # print(download_command)
            # subprocess.run(
            #     download_command,
            #     check=True,
            #     shell=True,
            # )

            # with open(download_save_path, "r") as file:
            #     dataset = [json.loads(line) for line in file]

            # print(f"Dataset size: {len(dataset)}")
            # num_per_split = len(dataset) / 64
            # print(f"Splitting into {num_per_split} samples per split")
            # dataset_chunk = [
            #     dataset[i : i + num_per_split]
            #     for i in range(0, len(dataset), num_per_split)
            # ]

            # dataset_paths = []
            # for i, chunk in enumerate(dataset_chunk):
            #     os.makedirs(f"{DATA_DIR}/{dataset_name}/raw_data/splits", exist_ok=True)
            #     _path = f"{DATA_DIR}/{dataset_name}/raw_data/splits/train_{i}.jsonl"
            #     dataset_paths.append(_path)
            #     with open(_path, "w+") as f:
            #         for line in chunk:
            #             f.write(json.dumps(line) + "\n")

            # del dataset_list
            # del dataset

            # processes = []
            # os.makedirs(f"{DATA_DIR}/{dataset_name}/generated_splits", exist_ok=True)
            # for i, dataset_path in enumerate(dataset_paths):

            #     output_dir = f"{DATA_DIR}/{dataset_name}/generated_splits/train_{i}.jsonl"

            #     generation_command = f"""
            #     nohup python gpt_recaption.py \
            #     --dataset_path {dataset_path} \
            #     --engine {engine} \
            #     --num_threads {num_threads} \
            #     --output_dir {output_dir} \
            #     --max_tokens {max_tokens} \
            #     --temperature {temperature} \
            #     --top_p {top_p} \
            #     --n {n} > {dataset_name}_{i}.out
            #     """

            #     print("Running generation command")
            #     print(generation_command)
            #     process = subprocess.Popen(generation_command, shell=True)

            #     # process.wait()
            #     processes.append(process)

            # for process in processes:
            #     process.wait()

            # os.makedirs(
            #     f"{DATA_DIR}/{dataset_name}",  # /{DATE}
            #     exist_ok=True,
            # )

            # count = 0
            # with open(
            #     f"{DATA_DIR}/{dataset_name}/train.jsonl",  # /{DATE}
            #     "w+",
            # ) as outfile:
            #     for split in os.listdir(f"{DATA_DIR}/{dataset_name}/generated_splits"):
            #         print(f"Merging {split}")
            #         with open(
            #             f"{DATA_DIR}/{dataset_name}/generated_splits/{split}", "r"
            #         ) as infile:
            #             for line in infile:
            #                 data = json.loads(line)
            #                 outfile.write(json.dumps(data) + "\n")
            #                 count += 1

            # print(f"Total samples merged and saved: {count}")
            # print(f"All files merged into {DATA_DIR}/{dataset_name}/train.jsonl")  # /{DATE}
            # shutil.rmtree(f"{DATA_DIR}/{dataset_name}/generated_splits")

            # shutil.rmtree(f"{DATA_DIR}/{dataset_name}/raw_data")

        # break


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
        default=96,
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

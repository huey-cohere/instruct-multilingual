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

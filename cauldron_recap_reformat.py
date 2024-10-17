import json
import os
from tqdm import tqdm
import multiprocessing as mp

FOLDER = "/home/olivernan_cohere_com/gpt_recaption_cauldron_2024_09_10_raw"
OUTPUT_BASE = "/home/olivernan_cohere_com/gpt_recaption_cauldron_2024_09_10_raw_reformat"
os.makedirs(OUTPUT_BASE, exist_ok=True)

def get_dir_list():
    dir_list = []
    for dataset in os.listdir(FOLDER):
        for version in os.listdir(os.path.join(FOLDER, dataset)):
            path = os.path.join(FOLDER, dataset, version)
            for filename in os.listdir(path):
                dir_list.append((dataset, os.path.join(path, filename)))
                
    return dir_list

# def get_dir_list():
#     dir_list = []
#     for dataset in os.listdir(FOLDER):
#         dir_list.append((dataset, os.path.join(FOLDER, dataset, "train.jsonl")))
                
#     return dir_list

# def rename_files():
#     for dataset in os.listdir(OUTPUT_BASE):
#         os.rename(os.path.join(OUTPUT_BASE, dataset), os.path.join(OUTPUT_BASE, dataset.replace("Rep", "Recap")))

def process_file(args):
    dataset, path = args
    output_folder = os.path.join(OUTPUT_BASE, "Recap"+dataset)
    os.makedirs(output_folder, exist_ok=True)
    
    output_file_path = os.path.join(output_folder, "train.jsonl")

    with open(path, "r") as input_file, open(output_file_path, "a") as output_file:
        for i, input_line in enumerate(input_file):
            data = json.loads(input_line)
            example = {"User": [], "Chatbot": [], "Image": []}
            example['command_id'] = data['command_id']
            try:
                example['metadata'] = data['metadata']
            except:
                if i == 1:
                    print(path)
                example['metadata'] = {"source": "open-llava-next"}
            example['index'] = i
            for turn in data["turns"]:
                if turn["role"] == "User":
                    for content in turn['content']:
                        if "text" in content:
                            example['User'].append({"text": content["text"], "language": "eng_Latn", "source": "raw"})
                        if "url" in content:
                            example['Image'].append(content["url"])
                elif turn["role"] == "Chatbot":
                    example['Chatbot'].append({'text': turn["content"], "language": "eng_Latn", "source": "raw"})
                elif turn["role"] == "Chatbot-gpt":
                    example['Chatbot'].append({"text": turn["content"], "language": "eng_Latn", "source": "raw-gpt_recap"})
            
            output_file.write(json.dumps(example) + "\n")

        
            

if __name__ == "__main__":
    dir_list = get_dir_list()
    print(f"Processing {len(dir_list)} files...")
    
    # # Clear existing output files
    # for dataset, _ in set((dataset, "") for dataset, _ in dir_list):
    #     output_folder = os.path.join(OUTPUT_BASE, dataset)
    #     os.makedirs(output_folder, exist_ok=True)
    #     open(os.path.join(output_folder, "train.jsonl"), "w").close()
    
    # Use all available CPU cores
    
    with mp.Pool(mp.cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(process_file, dir_list), total=len(dir_list)))

print("Processing complete.")